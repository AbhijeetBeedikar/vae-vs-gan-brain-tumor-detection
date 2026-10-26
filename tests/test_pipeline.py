"""Smoke test: train and evaluate every model end-to-end on tiny synthetic data.

The synthetic "brains" are ellipses with some internal structure; the
"tumours" are bright blobs added to the MSD-like set. This checks that all
pieces (splits, training, anomaly maps, metrics, figures) fit together
without needing the real datasets.
"""

import numpy as np
import pytest

from brainad.config import load_config
from brainad.data import load_split
from brainad.evaluate import IntensityBaseline, evaluate_model
from brainad.train import load_trained, train_model


def _fake_dataset(path, n_subjects, prefix, tumours, rng, size=128, slices=6):
    yy, xx = np.mgrid[:size, :size] / size - 0.5
    brain = (xx / 0.38) ** 2 + (yy / 0.45) ** 2 < 1
    images, brains, labels, subj, zs = [], [], [], [], []
    for s in range(n_subjects):
        for z in range(slices):
            img = 0.35 + 0.1 * np.sin(12 * xx + z) * np.cos(9 * yy + s) + rng.normal(0, 0.02, brain.shape)
            lab = np.zeros_like(brain)
            if tumours and z % 2 == 0:
                cx, cy = rng.uniform(-0.15, 0.15, 2)
                lab = ((xx - cx) ** 2 + (yy - cy) ** 2 < 0.08**2) & brain
                img = np.where(lab, 0.9, img)
            images.append(np.clip(img, 0, 1) * brain)
            brains.append(brain)
            labels.append(lab)
            subj.append(s)
            zs.append(z)
    np.savez(path, image=np.array(images, np.float16), brain=np.array(brains), label=np.array(labels),
             subject=np.array(subj, np.int32), z=np.array(zs, np.int16),
             subjects=np.array([f"{prefix}{i:03d}" for i in range(n_subjects)]),
             ncc=np.ones(n_subjects, np.float32))


@pytest.fixture()
def tiny_cfg(tmp_path):
    rng = np.random.default_rng(0)
    processed = tmp_path / "processed"
    processed.mkdir()
    _fake_dataset(processed / "ixi.npz", 20, "IXI", tumours=False, rng=rng)
    _fake_dataset(processed / "msd.npz", 20, "BRATS_", tumours=True, rng=rng)
    return {
        "paths": {"processed_dir": str(processed), "results_dir": str(tmp_path / "results")},
        "training": {"epochs": 1, "gan_epochs": 1, "encoder_epochs": 1, "batch_size": 16, "num_threads": 2},
        "model": {"latent_dim": 16, "base_channels": 8},
        "splits": {"msd_val_fraction": 0.2},
    }


@pytest.mark.parametrize("model_name", ["ae", "vae", "fanogan"])
def test_train_and_evaluate(model_name, tiny_cfg, tmp_path):
    cfg = load_config(f"configs/{model_name}.yaml", tiny_cfg)
    run_dir = tmp_path / "runs" / model_name
    train_model(cfg, load_split(cfg, "ixi_train"), load_split(cfg, "ixi_val"), run_dir)
    assert (run_dir / "model.pt").exists() and (run_dir / "val_reconstructions.png").exists()

    model, saved_cfg = load_trained(run_dir, "cpu")
    results = evaluate_model(model, saved_cfg, model_name, tmp_path / "results" / model_name)
    for residual in ("positive", "abs"):
        m = results[residual]
        assert 0.0 <= m["dice_global"] <= 1.0
        assert 0.0 <= m["pixel_auroc"] <= 1.0
    assert (tmp_path / "results" / model_name / "examples.png").exists()


def test_intensity_baseline_detects_bright_blobs(tiny_cfg, tmp_path):
    cfg = load_config("configs/base.yaml", tiny_cfg)
    results = evaluate_model(IntensityBaseline(), cfg, "intensity", tmp_path / "results" / "intensity")
    # Synthetic tumours are much brighter than the background tissue.
    assert results["positive"]["pixel_auroc"] > 0.95
