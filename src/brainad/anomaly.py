"""From reconstructions to anomaly maps, slice scores and tumour masks.

For each test slice ``x`` the model produces a healthy-looking
reconstruction ``x_hat``. The post-processing (identical for every model,
and reported explicitly as the proposal promises) is:

1. Residual:  r = max(x - x_hat, 0)   ('positive', default)
              r = |x - x_hat|         ('abs')
   Gliomas and their surrounding swelling are *bright* on T2, so only
   "brighter than expected" pixels are evidence of a tumour; the positive
   residual ignores places where the model merely over-predicts brightness.
2. Light smoothing with a Gaussian (sigma in pixels) to suppress
   single-pixel noise.
3. Keep only pixels inside the brain mask, shrunk ("eroded") by a few
   pixels, because reconstructions are least accurate at the brain border.
4. Threshold -> predicted tumour mask (threshold chosen on a small labelled
   validation subset, see :mod:`brainad.metrics`).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur (N,1,H,W) images with a Gaussian of standard deviation ``sigma`` pixels."""
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    # A 2D Gaussian is separable: blur along rows, then along columns.
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel_1d.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel_1d.view(1, 1, -1, 1))
    return x


def erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Shrink a binary (N,1,H,W) mask by ``pixels`` (min-filter = -maxpool(-x))."""
    if pixels <= 0:
        return mask
    k = 2 * pixels + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pixels)


def anomaly_maps(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    brain: torch.Tensor,
    residual: str = "positive",
    sigma: float = 1.0,
    erosion: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute anomaly maps.

    Returns:
        maps: (N,1,H,W) anomaly scores, 0 outside the evaluation mask.
        eval_mask: (N,1,H,W) the eroded brain mask where scores are valid.
    """
    if residual == "positive":
        r = (x - x_hat).clamp(min=0)
    elif residual == "abs":
        r = (x - x_hat).abs()
    else:
        raise ValueError(f"unknown residual type {residual!r}")
    r = gaussian_blur(r * brain, sigma)  # zero outside the brain first so nothing bleeds in
    eval_mask = erode(brain, erosion)
    return r * eval_mask, eval_mask


def slice_scores(maps: torch.Tensor, eval_mask: torch.Tensor, top_percent: float = 1.0) -> torch.Tensor:
    """One "does this slice contain a tumour?" score per slice.

    Score = mean of the highest ``top_percent`` % of anomaly values inside the
    brain. Using the top values (not the mean of the whole slice) means a small
    bright tumour is not diluted by the large healthy remainder.
    """
    n = maps.shape[0]
    flat = maps.view(n, -1).clone()
    inside = eval_mask.view(n, -1) > 0
    flat[~inside] = float("-inf")  # outside pixels sort to the end
    k = (inside.sum(1).float() * top_percent / 100).ceil().clamp(min=1).long()
    sorted_vals = flat.sort(dim=1, descending=True).values.clamp(min=0)  # -inf -> 0 (never inside top-k)
    csum = sorted_vals.cumsum(dim=1)
    top_sum = csum.gather(1, (k - 1).view(-1, 1)).squeeze(1)
    scores = top_sum / k
    scores[inside.sum(1) == 0] = 0.0  # slices with no brain pixels
    return scores
