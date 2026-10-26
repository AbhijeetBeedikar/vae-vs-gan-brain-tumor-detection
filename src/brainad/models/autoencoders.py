"""Plain autoencoder (baseline) and variational autoencoder.

Both are trained only on healthy slices to reconstruct their input. At test
time a tumour is something the model never learned to draw, so the
reconstruction looks "healthy" and differs from the input where the tumour is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .networks import Decoder, Encoder, init_weights


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Sum of squared errors per image, averaged over the batch."""
    return F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]


class AutoEncoder(nn.Module):
    """Deterministic autoencoder: x -> z (latent_dim numbers) -> x_hat.

    The narrow bottleneck forces the network to keep only the typical
    structure of a brain; it has no probabilistic latent space.
    """

    def __init__(self, image_size: int, latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.encoder = Encoder(image_size, latent_dim, base_channels)
        self.decoder = Decoder(image_size, latent_dim, base_channels)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        rec = reconstruction_error(x, self(x))
        return rec, {"loss": rec.item(), "recon": rec.item()}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


class VAE(nn.Module):
    """Variational autoencoder (Kingma & Welling, 2014).

    The encoder outputs a mean ``mu`` and log-variance ``logvar`` of a Gaussian
    over the latent code. During training we sample ``z = mu + sigma * eps``
    (the "reparameterisation trick", which keeps sampling differentiable) and
    minimise the negative evidence lower bound (ELBO):

        loss = reconstruction error + beta * KL( q(z|x) || N(0, I) )

    The KL term pulls codes toward a standard normal, giving a smooth,
    well-organised latent space.
    """

    def __init__(self, image_size: int, latent_dim: int = 128, base_channels: int = 32, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        # One encoder head produces both mu and logvar (2 * latent_dim numbers).
        self.encoder = Encoder(image_size, 2 * latent_dim, base_channels)
        self.decoder = Decoder(image_size, latent_dim, base_channels)
        self.apply(init_weights)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)  # clamp for numerical stability

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # reparameterisation trick
        return self.decoder(z), mu, logvar

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x_hat, mu, logvar = self(x)
        rec = reconstruction_error(x, x_hat)
        # Closed-form KL divergence between N(mu, sigma^2) and N(0, 1), summed over dims.
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
        total = rec + self.beta * kl
        return total, {"loss": total.item(), "recon": rec.item(), "kl": kl.item()}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """At test time decode the *mean* code (no sampling) for a deterministic map."""
        mu, _ = self.encode(x)
        return self.decoder(mu)
