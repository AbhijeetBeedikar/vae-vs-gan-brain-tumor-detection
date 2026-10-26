"""Training loops for the three models.

All models train only on *healthy* IXI slices. The IXI validation split is
used to keep the checkpoint with the lowest reconstruction error, so no
tumour data influences training.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .data import SliceSet, iterate_batches
from .models import build_model
from .utils import get_device, save_json, set_seed
from .visualize import save_reconstruction_grid, save_sample_grid


@torch.no_grad()
def reconstruct_all(model, images: torch.Tensor, batch_size: int = 256, device="cpu") -> torch.Tensor:
    """Reconstruct a whole tensor of images in batches (model in eval mode)."""
    model.eval()
    outs = [model.reconstruct(b.to(device)).cpu() for b in iterate_batches(images, batch_size, shuffle=False)]
    return torch.cat(outs)


def validation_error(model, images: torch.Tensor, device) -> float:
    """Mean per-pixel squared reconstruction error on validation images."""
    x_hat = reconstruct_all(model, images, device=device)
    return torch.mean((x_hat - images) ** 2).item()


def _log(history: list, entry: dict, start: float) -> None:
    entry["minutes"] = round((time.time() - start) / 60, 2)
    history.append(entry)
    print("  ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()), flush=True)


def _average(logs: list[dict]) -> dict:
    return {k: sum(d[k] for d in logs) / len(logs) for k in logs[0]}


def train_autoencoder(model, cfg: dict, train: SliceSet, val: SliceSet, run_dir: Path, device) -> list:
    """Standard training loop for the AE and the VAE."""
    tc = cfg["training"]
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    history, best, start = [], float("inf"), time.time()

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        logs = []
        for x in iterate_batches(train.image, tc["batch_size"], shuffle=True, hflip=tc["hflip"], generator=gen):
            loss, info = model.loss(x.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            logs.append(info)

        val_mse = validation_error(model, val.image, device)
        _log(history, {"epoch": epoch, **_average(logs), "val_mse": val_mse}, start)
        if val_mse < best:  # keep the checkpoint that reconstructs healthy brains best
            best = val_mse
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, run_dir / "model.pt")
    return history


def train_fanogan(model, cfg: dict, train: SliceSet, val: SliceSet, run_dir: Path, device) -> list:
    """Two-stage f-AnoGAN training (see :mod:`brainad.models.fanogan`)."""
    tc, mc = cfg["training"], cfg["model"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    history, start = [], time.time()

    # ---------------- Stage 1: WGAN-GP ----------------
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=tc["gan_lr"], betas=tuple(tc["gan_betas"]))
    opt_d = torch.optim.Adam(model.critic.parameters(), lr=tc["gan_lr"], betas=tuple(tc["gan_betas"]))
    step = 0
    for epoch in range(1, tc["gan_epochs"] + 1):
        model.generator.train()
        model.critic.train()
        d_logs, g_logs = [], []
        for x in iterate_batches(train.image, tc["batch_size"], shuffle=True, hflip=tc["hflip"], generator=gen):
            x = x.to(device)
            # Critic update on every batch ...
            d_loss, info = model.critic_loss(x)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            d_logs.append(info)
            # ... generator update once every n_critic batches.
            step += 1
            if step % mc["n_critic"] == 0:
                g_loss, info = model.generator_loss(tc["batch_size"], device)
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_logs.append(info)
        _log(history, {"stage": "gan", "epoch": epoch, **_average(d_logs), **_average(g_logs)}, start)

    model.generator.eval()
    with torch.no_grad():
        save_sample_grid(model.generator(model.sample_z(32, device)).cpu(), run_dir / "gan_samples.png")

    # ---------------- Stage 2: izi_f encoder (G and D frozen) ----------------
    for p in list(model.generator.parameters()) + list(model.critic.parameters()):
        p.requires_grad_(False)
    model.critic.eval()
    opt_e = torch.optim.Adam(model.encoder.parameters(), lr=tc["lr"])
    best = float("inf")
    for epoch in range(1, tc["encoder_epochs"] + 1):
        model.encoder.train()
        logs = []
        for x in iterate_batches(train.image, tc["batch_size"], shuffle=True, hflip=tc["hflip"], generator=gen):
            loss, info = model.encoder_loss(x.to(device))
            opt_e.zero_grad()
            loss.backward()
            opt_e.step()
            logs.append(info)
        val_mse = validation_error(model, val.image, device)
        _log(history, {"stage": "encoder", "epoch": epoch, **_average(logs), "val_mse": val_mse}, start)
        if val_mse < best:
            best = val_mse
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, run_dir / "model.pt")
    return history


def train_model(cfg: dict, train: SliceSet, val: SliceSet, run_dir: str | Path):
    """Build, train and save a model; returns the best checkpoint loaded into the model."""
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["training"].get("num_threads", torch.get_num_threads()))
    device = get_device()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training {cfg['model']['name']} ({n_params / 1e6:.2f}M parameters) on {len(train)} slices, device={device}")

    trainer = train_fanogan if cfg["model"]["name"] == "fanogan" else train_autoencoder
    history = trainer(model, cfg, train, val, run_dir, device)
    save_json(history, run_dir / "history.json")

    # Reload the best checkpoint and save example reconstructions of healthy validation slices.
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["state_dict"])
    examples = val.image[torch.linspace(0, len(val) - 1, 8).long()]
    save_reconstruction_grid(examples, reconstruct_all(model, examples, device=device), run_dir / "val_reconstructions.png")
    return model


def load_trained(run_dir: str | Path, device=None):
    """Load a trained model (and its config) from ``run_dir/model.pt``."""
    device = device or get_device()
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["config"]
