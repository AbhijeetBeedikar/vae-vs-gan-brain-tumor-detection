"""Model zoo: ``build_model(cfg)`` returns the network described by ``cfg['model']``."""

from __future__ import annotations

from torch import nn

from .autoencoders import VAE, AutoEncoder
from .fanogan import FAnoGAN

MODELS = {"ae": AutoEncoder, "vae": VAE, "fanogan": FAnoGAN}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model from the config (all extra keys become constructor arguments)."""
    params = dict(cfg["model"])
    name = params.pop("name")
    return MODELS[name](image_size=cfg["data"]["image_size"], **params)


__all__ = ["AutoEncoder", "VAE", "FAnoGAN", "build_model", "MODELS"]
