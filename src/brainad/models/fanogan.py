"""f-AnoGAN (Schlegl et al., 2019): fast GAN-based anomaly detection.

Training happens in two separate stages:

Stage 1 - a Wasserstein GAN with gradient penalty (WGAN-GP) learns to
    generate healthy slices from random noise z ~ N(0, I).
    * The critic D scores how "real" an image looks.
    * The generator G tries to make D score its fakes highly.
    * The gradient penalty keeps D smooth (1-Lipschitz), which makes GAN
      training much more stable than the original GAN loss.

Stage 2 - an encoder E learns to map an image to the latent code that G
    would need to redraw it ("izi_f" architecture), with G and D frozen:

        loss = ||x - G(E(x))||^2  +  kappa * ||f(x) - f(G(E(x)))||^2

    where f(.) are the critic's intermediate features. This replaces
    AnoGAN's slow per-image latent search with a single forward pass.

At test time the reconstruction of x is G(E(x)). Because G has only ever
produced healthy anatomy, a tumour is "repaired" in the reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .networks import Decoder, Encoder, init_weights


class FAnoGAN(nn.Module):
    def __init__(self, image_size: int, latent_dim: int = 128, base_channels: int = 32,
                 gp_weight: float = 10.0, kappa: float = 1.0, **_):
        super().__init__()
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.kappa = kappa
        self.generator = Decoder(image_size, latent_dim, base_channels)
        self.critic = Encoder(image_size, 1, base_channels, norm="layer")  # outputs one realness score
        self.encoder = Encoder(image_size, latent_dim, base_channels)
        self.apply(init_weights)

    # ----- Stage 1: WGAN-GP -------------------------------------------------

    def sample_z(self, n: int, device) -> torch.Tensor:
        return torch.randn(n, self.latent_dim, device=device)

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Penalise the critic when its gradient norm (w.r.t. the input) differs from 1,
        evaluated at random points on lines between real and fake images."""
        alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
        mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        scores = self.critic(mixed)
        grads, = torch.autograd.grad(scores.sum(), mixed, create_graph=True)
        return ((grads.flatten(1).norm(dim=1) - 1) ** 2).mean()

    def critic_loss(self, real: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Critic wants high scores for real and low scores for fake images."""
        with torch.no_grad():
            fake = self.generator(self.sample_z(real.shape[0], real.device))
        wasserstein = self.critic(real).mean() - self.critic(fake).mean()
        gp = self.gradient_penalty(real, fake)
        loss = -wasserstein + self.gp_weight * gp
        return loss, {"d_loss": loss.item(), "wasserstein": wasserstein.item(), "gp": gp.item()}

    def generator_loss(self, batch_size: int, device) -> tuple[torch.Tensor, dict]:
        """Generator wants the critic to give its fakes high scores."""
        fake = self.generator(self.sample_z(batch_size, device))
        loss = -self.critic(fake).mean()
        return loss, {"g_loss": loss.item()}

    # ----- Stage 2: izi_f encoder ---------------------------------------------

    def encoder_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Image-space plus critic-feature-space reconstruction error (G and D frozen)."""
        x_hat = self.generator(self.encoder(x))
        _, feat_real = self.critic(x, return_features=True)
        _, feat_fake = self.critic(x_hat, return_features=True)
        img_loss = F.mse_loss(x_hat, x)
        feat_loss = F.mse_loss(feat_fake, feat_real)
        loss = img_loss + self.kappa * feat_loss
        return loss, {"loss": loss.item(), "img": img_loss.item(), "feat": feat_loss.item()}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(self.encoder(x))
