"""DCGAN-style building blocks shared by all three models.

Following Radford et al. (2016), the networks use only strided
convolutions (no pooling): every encoder layer halves the image size with a
4x4 convolution of stride 2, and every decoder layer doubles it with a 4x4
transposed convolution. For a 64x64 input:

    Encoder: 1x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8 -> 256x4x4 -> linear -> out_dim
    Decoder: latent -> linear -> 256x4x4 -> 128x8x8 -> 64x16x16 -> 32x32x32 -> 1x64x64

A 128x128 input simply gets one more layer on each side.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def num_layers(image_size: int) -> int:
    """How many stride-2 layers take ``image_size`` down to 4x4."""
    n = int(math.log2(image_size)) - 2
    if 2 ** (n + 2) != image_size:
        raise ValueError(f"image_size must be a power of two >= 8, got {image_size}")
    return n


def _norm(kind: str, channels: int) -> nn.Module:
    """Normalisation layer. WGAN-GP critics must not use batch norm (the gradient
    penalty is computed per sample), so they use layer norm instead."""
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "layer":
        return nn.GroupNorm(1, channels)  # GroupNorm with 1 group == LayerNorm over C,H,W
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


def _channels(n_layers: int, base: int, max_channels: int = 512) -> list[int]:
    """Channel width after each layer: base, 2*base, 4*base, ... (capped)."""
    return [min(base * 2**i, max_channels) for i in range(n_layers)]


class Encoder(nn.Module):
    """Image (B,1,H,W) -> vector (B,out_dim). Also used as the GAN critic."""

    def __init__(self, image_size: int, out_dim: int, base_channels: int = 32, norm: str = "batch"):
        super().__init__()
        chans = _channels(num_layers(image_size), base_channels)
        layers, in_ch = [], 1
        for i, ch in enumerate(chans):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1, bias=(norm == "none" or i == 0)))
            if i > 0:  # DCGAN: no normalisation on the first layer
                layers.append(_norm(norm, ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.feature_dim = in_ch * 4 * 4
        self.fc = nn.Linear(self.feature_dim, out_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.features(x).flatten(1)
        out = self.fc(h)
        return (out, h) if return_features else out


class Decoder(nn.Module):
    """Latent vector (B,latent_dim) -> image (B,1,H,W) with values in [0, 1]."""

    def __init__(self, image_size: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        chans = _channels(num_layers(image_size), base_channels)[::-1]  # widest first
        self.start_channels = chans[0]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, chans[0] * 4 * 4),
            nn.BatchNorm1d(chans[0] * 4 * 4),
            nn.ReLU(inplace=True),
        )
        layers = []
        for in_ch, out_ch in zip(chans[:-1], chans[1:]):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        # Last layer to a single grey-scale channel; sigmoid keeps pixels in [0, 1]
        # like our normalised MRI slices.
        layers += [nn.ConvTranspose2d(chans[-1], 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.start_channels, 4, 4)
        return self.net(h)


def init_weights(module: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) for conv weights, N(1, 0.02) for norm scales."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
