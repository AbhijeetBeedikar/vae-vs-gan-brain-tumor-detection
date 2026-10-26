"""Evaluation metrics.

* Dice overlap between predicted and expert tumour masks
  (2|P ∩ G| / (|P| + |G|); 1 = perfect, 0 = no overlap).
* Pixel-level AUROC and AUPRC: how well the raw anomaly scores rank tumour
  pixels above healthy pixels, without choosing a threshold. AUPRC is the
  more informative of the two when tumour pixels are rare.
* Slice-level AUROC: can the slice score tell slices with a tumour from
  slices without one?
* False-positive rate on healthy IXI test slices: the fraction of healthy
  brain pixels wrongly flagged at the chosen threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice coefficient of two boolean arrays (1.0 if both are empty)."""
    pred, target = pred.astype(bool), target.astype(bool)
    denom = pred.sum() + target.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, target).sum() / denom)


def best_dice_threshold(scores: np.ndarray, labels: np.ndarray, n_bins: int = 2000) -> tuple[float, float]:
    """Threshold that maximises Dice on a labelled validation set.

    Instead of trying every threshold one at a time, we histogram the scores
    of tumour and non-tumour pixels separately. A cumulative sum from the top
    bin down then gives, for every candidate threshold at once, the number of
    true positives (TP) and false positives (FP):

        Dice(t) = 2 TP(t) / (TP(t) + FP(t) + #tumour pixels)

    Args:
        scores: anomaly scores of the evaluated pixels (1-D).
        labels: matching boolean tumour labels (1-D).
    Returns:
        (threshold, dice at that threshold)
    """
    scores = scores.astype(np.float64)
    labels = labels.astype(bool)
    hi = np.quantile(scores, 0.9999) if scores.size else 1.0
    edges = np.linspace(0.0, max(hi, 1e-8), n_bins + 1)
    pos_hist, _ = np.histogram(scores[labels], bins=edges)
    neg_hist, _ = np.histogram(scores[~labels], bins=edges)
    # Pixels above the top edge belong to the last bin for "score >= edge".
    pos_hist[-1] += np.sum(scores[labels] > edges[-1])
    neg_hist[-1] += np.sum(scores[~labels] > edges[-1])
    tp = np.cumsum(pos_hist[::-1])[::-1]  # tp[i] = #tumour pixels with score >= edges[i]
    fp = np.cumsum(neg_hist[::-1])[::-1]
    dices = 2 * tp / np.maximum(tp + fp + labels.sum(), 1)
    best = int(np.argmax(dices))
    return float(edges[best]), float(dices[best])


def pixel_auroc_auprc(scores: np.ndarray, labels: np.ndarray, max_pixels: int = 20_000_000, seed: int = 0) -> tuple[float, float]:
    """Pixel-level AUROC and AUPRC (average precision).

    If there are more than ``max_pixels`` pixels, a uniform random subset is
    used to keep memory and time manageable (this does not change the
    expected value of either metric).
    """
    if scores.size > max_pixels:
        idx = np.random.default_rng(seed).choice(scores.size, max_pixels, replace=False)
        scores, labels = scores[idx], labels[idx]
    labels = labels.astype(bool)
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def slice_auroc(scores: np.ndarray, has_tumour: np.ndarray) -> float:
    """AUROC of slice scores for the binary question "does this slice contain a tumour?"."""
    has_tumour = has_tumour.astype(bool)
    if has_tumour.all() or not has_tumour.any():
        return float("nan")
    return float(roc_auc_score(has_tumour, scores))


def false_positive_rate(pred: np.ndarray) -> float:
    """Fraction of (healthy) pixels predicted as tumour."""
    return float(pred.mean()) if pred.size else float("nan")
