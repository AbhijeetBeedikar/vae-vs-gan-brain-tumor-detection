"""Loading preprocessed slices and making subject-level splits.

Splits are made *by subject* so no person's brain appears in more than one
split (otherwise near-identical neighbouring slices would leak between
train and test and inflate the results).

    IXI (healthy):  80% train / 10% validation / 10% healthy test
    MSD (tumour):   10% "threshold validation" / 90% test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .utils import load_json, save_json


@dataclass
class SliceSet:
    """A set of 2D slices kept in memory as tensors.

    image/brain/label have shape (N, 1, H, W); subject and z have shape (N,).
    """

    image: torch.Tensor
    brain: torch.Tensor
    label: torch.Tensor
    subject: np.ndarray
    z: np.ndarray
    subject_names: np.ndarray

    def __len__(self) -> int:
        return len(self.image)


def _downsample(x: np.ndarray, size: int, binary: bool) -> torch.Tensor:
    """(N, S, S) array -> (N, 1, size, size) tensor via area averaging."""
    t = torch.from_numpy(x.astype(np.float32))[:, None]
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="area")
    return (t > 0.5).float() if binary else t


def load_slices(npz_path: str | Path, image_size: int, subjects: list[str] | None = None) -> SliceSet:
    """Load a processed dataset, optionally keeping only some subjects.

    Slices are stored at 128x128 and resized here, so the same preprocessed
    file serves both the 64x64 and the 128x128 experiments.
    """
    d = np.load(npz_path)
    names = d["subjects"]
    keep = np.ones(len(d["subject"]), dtype=bool)
    if subjects is not None:
        wanted = np.isin(names, np.asarray(subjects))
        keep = wanted[d["subject"]]

    image = _downsample(d["image"][keep], image_size, binary=False)
    brain = _downsample(d["brain"][keep], image_size, binary=True)
    label = _downsample(d["label"][keep], image_size, binary=True) * brain
    image = image * brain  # resized pixels at the edge must not leak outside the mask
    return SliceSet(image, brain, label, d["subject"][keep], d["z"][keep], names)


def make_splits(processed_dir: str | Path, cfg: dict) -> dict:
    """Create (or load, if already saved) the subject-level splits.

    The splits are written to ``processed_dir/splits.json`` the first time,
    so every model is trained and tested on exactly the same subjects.
    """
    path = Path(processed_dir) / "splits.json"
    if path.exists():
        return load_json(path)

    rng = np.random.default_rng(cfg["seed"])

    ixi = sorted(np.load(Path(processed_dir) / "ixi.npz")["subjects"].tolist())
    rng.shuffle(ixi)
    f_train, f_val, _ = cfg["splits"]["ixi_fractions"]
    n_train = int(round(f_train * len(ixi)))
    n_val = int(round(f_val * len(ixi)))

    msd = sorted(np.load(Path(processed_dir) / "msd.npz")["subjects"].tolist())
    rng.shuffle(msd)
    n_msd_val = int(round(cfg["splits"]["msd_val_fraction"] * len(msd)))

    splits = {
        "ixi_train": sorted(ixi[:n_train]),
        "ixi_val": sorted(ixi[n_train : n_train + n_val]),
        "ixi_test": sorted(ixi[n_train + n_val :]),
        "msd_val": sorted(msd[:n_msd_val]),
        "msd_test": sorted(msd[n_msd_val:]),
    }
    save_json(splits, path)
    return splits


def load_split(cfg: dict, split: str) -> SliceSet:
    """Load one named split, e.g. ``"ixi_train"`` or ``"msd_test"``."""
    processed = Path(cfg["paths"]["processed_dir"])
    splits = make_splits(processed, cfg)
    dataset = "ixi" if split.startswith("ixi") else "msd"
    return load_slices(processed / f"{dataset}.npz", cfg["data"]["image_size"], splits[split])


def iterate_batches(images: torch.Tensor, batch_size: int, shuffle: bool, hflip: bool = False, generator=None):
    """Yield mini-batches from an in-memory tensor.

    The whole dataset fits in RAM, so this is simpler and faster than a
    DataLoader. With ``hflip`` each image is mirrored left-right with
    probability 0.5 (data augmentation).
    """
    n = len(images)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch = images[order[start : start + batch_size]]
        if hflip:
            flip = torch.rand(len(batch), generator=generator) < 0.5
            batch = batch.clone()
            batch[flip] = batch[flip].flip(-1)
        yield batch
