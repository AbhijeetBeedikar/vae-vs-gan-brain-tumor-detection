"""Fast unit tests for the building blocks (models, anomaly maps, metrics, config)."""

import numpy as np
import pytest
import torch

from brainad.anomaly import anomaly_maps, erode, gaussian_blur, slice_scores
from brainad.config import deep_merge, parse_override
from brainad.metrics import best_dice_threshold, dice, pixel_auroc_auprc
from brainad.models import build_model


@pytest.mark.parametrize("name", ["ae", "vae", "fanogan"])
@pytest.mark.parametrize("size", [64, 128])
def test_models_shapes_and_losses(name, size):
    cfg = {"data": {"image_size": size}, "model": {"name": name, "latent_dim": 16, "base_channels": 8}}
    model = build_model(cfg)
    x = torch.rand(4, 1, size, size)
    assert model.reconstruct(x).shape == x.shape
    if name == "fanogan":
        losses = [model.critic_loss(x)[0], model.generator_loss(4, "cpu")[0], model.encoder_loss(x)[0]]
    else:
        losses = [model.loss(x)[0]]
    for loss in losses:
        assert torch.isfinite(loss)
        loss.backward()


def test_reconstruction_in_unit_range():
    model = build_model({"data": {"image_size": 64}, "model": {"name": "ae", "latent_dim": 8, "base_channels": 8}})
    out = model.reconstruct(torch.rand(2, 1, 64, 64))
    assert out.min() >= 0 and out.max() <= 1


def test_dice():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    assert dice(a, a) == 1.0
    assert dice(a, ~a) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0


def test_best_threshold_finds_separating_value():
    rng = np.random.default_rng(0)
    labels = rng.random(10_000) < 0.1
    scores = np.where(labels, 0.8, 0.2) + rng.normal(0, 0.05, labels.size)
    thr, d = best_dice_threshold(scores, labels)
    assert 0.35 < thr < 0.65
    assert d > 0.99


def test_pixel_auroc_perfect_and_random():
    labels = np.array([0, 0, 1, 1])
    assert pixel_auroc_auprc(np.array([0.1, 0.2, 0.8, 0.9]), labels)[0] == 1.0
    assert pixel_auroc_auprc(np.array([0.9, 0.8, 0.2, 0.1]), labels)[0] == 0.0


def test_positive_residual_ignores_dark_errors():
    x = torch.zeros(1, 1, 32, 32)
    x_hat = torch.zeros_like(x)
    x[..., 10:14, 10:14] = 1.0      # bright "tumour" -> should be flagged
    x_hat[..., 20:24, 20:24] = 1.0  # model over-predicts brightness -> ignored by 'positive'
    brain = torch.ones_like(x)
    pos, _ = anomaly_maps(x, x_hat, brain, "positive", sigma=0, erosion=0)
    ab, _ = anomaly_maps(x, x_hat, brain, "abs", sigma=0, erosion=0)
    assert pos[..., 20:24, 20:24].sum() == 0 and pos[..., 10:14, 10:14].sum() > 0
    assert ab[..., 20:24, 20:24].sum() > 0


def test_blur_preserves_mass_and_erosion_shrinks():
    x = torch.zeros(1, 1, 31, 31)
    x[..., 15, 15] = 1.0
    assert torch.isclose(gaussian_blur(x, 1.5).sum(), torch.tensor(1.0), atol=1e-5)
    m = torch.zeros(1, 1, 20, 20)
    m[..., 5:15, 5:15] = 1
    assert erode(m, 2).sum() == 36  # 10x10 square -> 6x6


def test_slice_scores_top_percent():
    maps = torch.zeros(2, 1, 10, 10)
    maps[0, 0, 0, 0] = 5.0
    mask = torch.ones_like(maps)
    s = slice_scores(maps, mask, top_percent=1.0)  # top 1% of 100 pixels = 1 pixel
    assert s[0] == 5.0 and s[1] == 0.0


def test_config_helpers():
    assert parse_override("training.epochs=5") == {"training": {"epochs": 5}}
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
