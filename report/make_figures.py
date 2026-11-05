"""Make the figures used in the midway report (report/figures/*.pdf).

Everything is read from the committed results/ folder, so this runs without
the datasets or the trained checkpoints:

    python report/make_figures.py

Figures:
    examples.pdf     test slices with each model's reconstruction and anomaly map
    curves.pdf       healthy validation error during training, plus the f-AnoGAN
                     Wasserstein estimate
    tumour_size.pdf  Dice per subject, grouped by tumour size

It also prints the numbers that the report text quotes for the size analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

MODELS = {"intensity": "Intensity", "ae": "AE", "vae": "VAE", "fanogan": "f-AnoGAN"}
COLORS = {"Intensity": "#8c8c8c", "AE": "#1f77b4", "VAE": "#ff7f0e", "f-AnoGAN": "#2ca02c"}

# Small fonts so single-column figures (3.25 in wide) stay legible but compact.
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def examples_figure(rows=(0, 3, 5)) -> None:
    """Full-width panel: input + truth, then reconstruction / anomaly map per model.

    The intensity baseline 'reconstructs' an empty image, so only its map is shown.
    Green = expert tumour outline, cyan = prediction at the validation-chosen threshold.
    """
    ex = {name: dict(np.load(RESULTS / f"{key}_64" / "examples.npz")) for key, name in MODELS.items()}
    cols = [("input + truth", "Intensity", None)]
    cols += [("Intensity: map", "Intensity", "map")]
    for name in ("AE", "VAE", "f-AnoGAN"):
        cols += [(f"{name}: recon", name, "recon"), (f"{name}: map", name, "map")]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6.9, 0.9 * len(rows) + 0.15))
    for r, i in enumerate(rows):
        for c, (title, name, kind) in enumerate(cols):
            ax = axes[r, c]
            e = ex[name]
            if kind is None:  # the input slice with the ground-truth outline
                ax.imshow(e["image"][i], cmap="gray", vmin=0, vmax=1)
                ax.contour(e["label"][i], levels=[0.5], colors="lime", linewidths=0.6)
            elif kind == "recon":
                ax.imshow(e["recon"][i], cmap="gray", vmin=0, vmax=1)
            else:  # anomaly map, scaled per model so the colour range is comparable
                ax.imshow(e["map"][i], cmap="inferno", vmin=0, vmax=np.percentile(e["map"], 99.9))
                ax.contour(e["pred"][i], levels=[0.5], colors="cyan", linewidths=0.5)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=6.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.03, hspace=0.03)
    fig.savefig(OUT / "examples.pdf")
    plt.close(fig)


def curves_figure() -> None:
    """Left: healthy validation MSE per epoch. Right: f-AnoGAN critic's Wasserstein estimate."""
    hist = {MODELS[k]: json.loads((RESULTS / f"{k}_64" / "history.json").read_text()) for k in ("ae", "vae", "fanogan")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.3, 1.45))
    for name, h in hist.items():
        val = [e for e in h if "val_mse" in e]  # f-AnoGAN: only the encoder stage has val_mse
        a1.plot([e["epoch"] for e in val], [e["val_mse"] for e in val], color=COLORS[name], label=name, lw=1)
    a1.set_yscale("log")
    a1.set_xlabel("epoch (f-AnoGAN: encoder stage)")
    a1.set_ylabel("healthy val. MSE")
    a1.legend(frameon=False)

    gan = [e for e in hist["f-AnoGAN"] if e.get("stage") == "gan"]
    a2.plot([e["epoch"] for e in gan], [e["wasserstein"] for e in gan], color=COLORS["f-AnoGAN"], lw=1)
    a2.set_xlabel("GAN epoch")
    a2.set_ylabel("Wasserstein estimate")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "curves.pdf")
    plt.close(fig)


def tumour_size_figure() -> None:
    """Mean Dice (± standard error) in three equal-count groups of tumour size."""
    per = {}
    for key, name in MODELS.items():
        with open(RESULTS / f"{key}_64" / "per_subject.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        per[name] = {r["subject"]: (float(r["dice"]), int(r["tumour_pixels"])) for r in rows}

    subjects = sorted(per["VAE"])
    size = np.array([per["VAE"][s][1] for s in subjects])
    cuts = np.quantile(size, [1 / 3, 2 / 3])
    group = np.digitize(size, cuts)  # 0 = small, 1 = medium, 2 = large
    labels = ["small", "medium", "large"]
    print(f"tumour-size tertile cut points (64x64 pixels, summed over slices): {cuts.round(0)}")

    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    width = 0.2
    for m, name in enumerate(MODELS.values()):
        dice = np.array([per[name][s][0] for s in subjects])
        means = [dice[group == g].mean() for g in range(3)]
        sems = [dice[group == g].std(ddof=1) / np.sqrt((group == g).sum()) for g in range(3)]
        x = np.arange(3) + (m - 1.5) * width
        ax.bar(x, means, width, yerr=sems, color=COLORS[name], label=name, error_kw={"lw": 0.6, "capsize": 1.5})
        print(f"{name:10s} Dice small/medium/large = " + " / ".join(f"{v:.3f}" for v in means)
              + f"   subjects with Dice < 0.2: {(dice < 0.2).mean():.1%}")
    ax.set_xticks(range(3), [f"{l} (n={(group == g).sum()})" for g, l in enumerate(labels)])
    ax.set_ylabel("Dice (subject mean)")
    ax.set_ylim(0, 0.8)
    ax.legend(ncol=4, frameon=False, loc="upper left", columnspacing=0.8, handlelength=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "tumour_size.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    examples_figure()
    curves_figure()
    tumour_size_figure()
    print(f"figures written to {OUT}")
