"""Evaluate trained models and write metrics + figures to results/.

Usage:
    python scripts/evaluate.py --runs runs/ae_64 runs/vae_64 runs/fanogan_64 --baseline

Writes results/<model>_<size>/{metrics.json, per_subject.csv, examples.png}
and, across all models, results/summary.md, results/summary.csv and comparison figures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainad.config import load_config  # noqa: E402
from brainad.evaluate import IntensityBaseline, evaluate_model  # noqa: E402
from brainad.train import load_trained  # noqa: E402
from brainad.utils import get_device, load_json  # noqa: E402
from brainad.visualize import save_comparison_panel, save_metric_bars, save_training_curves  # noqa: E402

# Metrics shown in the summary table: (key, column title).
TABLE = [
    ("dice_subject_mean", "Dice (subject mean)"),
    ("dice_global", "Dice (global)"),
    ("pixel_auroc", "Pixel AUROC"),
    ("pixel_auprc", "Pixel AUPRC"),
    ("slice_auroc_msd", "Slice AUROC (MSD)"),
    ("slice_auroc_vs_healthy", "Slice AUROC (vs IXI)"),
    ("healthy_fpr", "Healthy pixel FPR"),
]
PRETTY = {"ae": "AE", "vae": "VAE", "fanogan": "f-AnoGAN", "intensity": "Intensity baseline"}


def write_summary(all_results: dict, results_dir: Path) -> None:
    """Markdown + CSV tables for both residual types, plus a bar chart."""
    lines = []
    for residual in ("positive", "abs"):
        lines.append(f"### Residual: `{residual}`" + (" (main setting)" if residual == "positive" else ""))
        lines.append("")
        lines.append("| Model | " + " | ".join(t for _, t in TABLE) + " |")
        lines.append("|---" * (len(TABLE) + 1) + "|")
        for name, res in all_results.items():
            m = res[residual]
            cells = []
            for key, _ in TABLE:
                if key == "dice_subject_mean":
                    cells.append(f"{m[key]:.3f} ± {m['dice_subject_std']:.3f}")
                elif key == "healthy_fpr":
                    cells.append(f"{100 * m[key]:.2f}%")
                else:
                    cells.append(f"{m[key]:.3f}")
            lines.append(f"| {PRETTY.get(name, name)} | " + " | ".join(cells) + " |")
        lines.append("")
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "residual"] + [k for k, _ in TABLE] + ["dice_subject_std", "threshold"])
        for name, res in all_results.items():
            for residual in ("positive", "abs"):
                m = res[residual]
                writer.writerow([name, residual] + [m[k] for k, _ in TABLE] + [m["dice_subject_std"], m["threshold"]])

    main = {PRETTY.get(n, n): r[r["main_residual"]] for n, r in all_results.items()}
    save_metric_bars(main, [k for k, _ in TABLE if k != "healthy_fpr"], results_dir / "metrics_bar.png")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="run directories containing model.pt")
    parser.add_argument("--baseline", action="store_true", help="also evaluate the intensity baseline")
    parser.add_argument("--config", default="configs/base.yaml", help="config for the baseline")
    args = parser.parse_args()

    device = get_device()
    all_results, examples, histories = {}, {}, {}
    results_dir = None

    jobs = []
    if args.baseline:
        jobs.append(("intensity", IntensityBaseline(), load_config(args.config), None))
    for run in args.runs:
        model, cfg = load_trained(run, device)
        jobs.append((cfg["model"]["name"], model, cfg, Path(run)))

    for name, model, cfg, run in jobs:
        results_dir = Path(cfg["paths"]["results_dir"])
        out_dir = results_dir / f"{name}_{cfg['data']['image_size']}"
        all_results[name] = evaluate_model(model, cfg, name, out_dir, device)
        examples[PRETTY[name]] = dict(np.load(out_dir / "examples.npz"))
        if run is not None:
            histories[PRETTY[name]] = load_json(run / "history.json")

    if results_dir is not None and len(all_results) > 1:
        write_summary(all_results, results_dir)
        save_comparison_panel(examples, results_dir / "comparison.png")
    if histories:
        save_training_curves(histories, results_dir / "training_curves.png")


if __name__ == "__main__":
    main()
