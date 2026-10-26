"""End-to-end evaluation of one trained model (or the intensity baseline).

Steps:
1. Reconstruct the healthy IXI test slices and the MSD validation/test slices.
2. Turn reconstructions into anomaly maps (:mod:`brainad.anomaly`).
3. Pick the pixel threshold that maximises Dice on the small MSD validation
   subset (the only place tumour labels are used before testing).
4. On MSD test: Dice (global and per subject), pixel AUROC/AUPRC and slice
   AUROC. On IXI test: the healthy false-positive rate.

Both residual types ('positive' and 'abs') are scored from the same
reconstructions so the effect of this post-processing choice is reported.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .anomaly import anomaly_maps, slice_scores
from .data import SliceSet, load_split
from .metrics import best_dice_threshold, dice, false_positive_rate, pixel_auroc_auprc, slice_auroc
from .train import reconstruct_all
from .utils import save_json
from .visualize import save_example_panel


class IntensityBaseline(torch.nn.Module):
    """Non-learned reference: "reconstruct" every slice as black.

    The positive residual is then just the image itself, i.e. "bright on T2 =
    tumour". Any useful generative model should beat this.
    """

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def example_indices(data: SliceSet, n: int = 6, seed: int = 0) -> np.ndarray:
    """Pick ``n`` test slices with clearly visible tumours, from different subjects.

    Deterministic, so every model is shown on exactly the same slices.
    """
    tumour_px = data.label.sum(dim=(1, 2, 3)).numpy()
    rng = np.random.default_rng(seed)
    chosen, used_subjects = [], set()
    for idx in rng.permutation(np.where(tumour_px > np.percentile(tumour_px[tumour_px > 0], 50))[0]):
        if data.subject[idx] not in used_subjects:
            chosen.append(idx)
            used_subjects.add(data.subject[idx])
        if len(chosen) == n:
            break
    return np.array(sorted(chosen))


def _maps(data: SliceSet, recon: torch.Tensor, ac: dict, residual: str):
    return anomaly_maps(data.image, recon, data.brain, residual, ac["smoothing_sigma"], ac["mask_erosion"])


def score_residual(cfg: dict, residual: str, sets: dict, recons: dict) -> tuple[dict, dict, list]:
    """All metrics for one residual type. Returns (metrics, test outputs, per-subject rows)."""
    ac = cfg["anomaly"]

    # --- 1. threshold on MSD validation subset ------------------------------
    val_map, val_mask = _maps(sets["msd_val"], recons["msd_val"], ac, residual)
    inside = val_mask.bool()
    threshold, val_dice = best_dice_threshold(val_map[inside].numpy(), sets["msd_val"].label.bool()[inside].numpy())

    # --- 2. MSD test -----------------------------------------------------------
    test = sets["msd_test"]
    test_map, test_mask = _maps(test, recons["msd_test"], ac, residual)
    inside = test_mask.bool()
    pred = (test_map >= threshold) & inside
    label = test.label.bool() & inside

    pix_auroc, pix_auprc = pixel_auroc_auprc(test_map[inside].numpy(), label[inside].numpy(), seed=cfg["seed"])

    s_scores = slice_scores(test_map, test_mask, ac["top_percent"]).numpy()
    has_tumour = label.sum(dim=(1, 2, 3)).numpy() >= ac["min_tumor_pixels"]

    # Per-subject Dice over each subject's slices (subjects with tumour in the kept slices).
    rows = []
    for s in np.unique(test.subject):
        sel = test.subject == s
        if label[sel].sum() > 0:
            rows.append({"subject": str(test.subject_names[s]), "dice": dice(pred[sel].numpy(), label[sel].numpy()),
                         "tumour_pixels": int(label[sel].sum())})

    # --- 3. healthy IXI test: false positives -----------------------------------
    healthy = sets["ixi_test"]
    h_map, h_mask = _maps(healthy, recons["ixi_test"], ac, residual)
    h_inside = h_mask.bool()
    h_pred = (h_map >= threshold) & h_inside
    h_scores = slice_scores(h_map, h_mask, ac["top_percent"]).numpy()

    # Slice-level AUROC, tumour slices of MSD vs. all healthy IXI slices.
    tumour_vs_healthy = slice_auroc(
        np.concatenate([s_scores[has_tumour], h_scores]),
        np.concatenate([np.ones(has_tumour.sum()), np.zeros(len(h_scores))]),
    )

    metrics = {
        "threshold": threshold,
        "val_dice": val_dice,
        "dice_global": dice(pred.numpy(), label.numpy()),
        "dice_subject_mean": float(np.mean([r["dice"] for r in rows])),
        "dice_subject_std": float(np.std([r["dice"] for r in rows])),
        "pixel_auroc": pix_auroc,
        "pixel_auprc": pix_auprc,
        "slice_auroc_msd": slice_auroc(s_scores, has_tumour),
        "slice_auroc_vs_healthy": tumour_vs_healthy,
        "healthy_fpr": false_positive_rate(h_pred[h_inside].numpy()),
        "healthy_slices_flagged": float((h_pred.sum(dim=(1, 2, 3)) >= ac["min_tumor_pixels"]).float().mean()),
        "tumour_prevalence": float(label[inside].float().mean()),
    }
    outputs = {"map": test_map, "pred": pred}
    return metrics, outputs, rows


def evaluate_model(model, cfg: dict, name: str, out_dir: str | Path, device="cpu") -> dict:
    """Evaluate a model and write metrics.json, per_subject.csv and figures to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = {s: load_split(cfg, s) for s in ("ixi_test", "msd_val", "msd_test")}
    recons = {s: reconstruct_all(model, d.image, device=device) for s, d in sets.items()}

    results = {"model": name, "image_size": cfg["data"]["image_size"], "main_residual": cfg["anomaly"]["residual"],
               "n_slices": {s: len(d) for s, d in sets.items()},
               "n_subjects": {s: int(len(np.unique(d.subject))) for s, d in sets.items()}}
    main_outputs, main_rows = None, None
    for residual in ("positive", "abs"):
        metrics, outputs, rows = score_residual(cfg, residual, sets, recons)
        results[residual] = metrics
        if residual == cfg["anomaly"]["residual"]:
            main_outputs, main_rows = outputs, rows
        print(f"[{name}] residual={residual}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    save_json(results, out_dir / "metrics.json")
    with open(out_dir / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "dice", "tumour_pixels"])
        writer.writeheader()
        writer.writerows(main_rows)

    # Qualitative examples (same slices for every model) for figures and the comparison panel.
    test = sets["msd_test"]
    idx = example_indices(test)
    examples = {
        "image": test.image[idx, 0].numpy(),
        "recon": recons["msd_test"][idx, 0].numpy(),
        "map": main_outputs["map"][idx, 0].numpy(),
        "pred": main_outputs["pred"][idx, 0].numpy(),
        "label": test.label[idx, 0].numpy(),
    }
    np.savez_compressed(out_dir / "examples.npz", **examples)
    save_example_panel(examples, out_dir / "examples.png", title=name)
    return results
