"""Turn raw 3D MRI volumes into aligned, normalised 2D axial slices.

The proposal requires the *same* pipeline for the healthy training data (IXI)
and the tumour test data (MSD), otherwise a model could learn to flag
"different hospital" instead of "tumour". For every scan we:

1. Load the T2 volume and reorient it to the standard RAS orientation.
2. Register (align) it to the MNI template with an affine transform.
3. Skull-strip: keep only voxels inside the template brain mask
   (for MSD, whose scans are already skull-stripped, we additionally
   intersect with the scan's own brain outline).
4. Normalise intensities: divide by the 99th percentile of brain voxels and
   clip to [0, 1], so brightness values are on the same scale.
5. Extract axial slices from the middle of the brain (skipping the
   near-empty top and bottom) and resize them to ``slice_size`` pixels.

Because step 2 puts every scan into the same space, slice number ``z`` shows
the same anatomy level in every subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .registration import apply_affine, ncc, register_affine, resample
from .template import Template, load_template

# Channel order of the 4D MSD BrainTumour images (see the task's dataset.json).
MSD_CHANNELS = {"FLAIR": 0, "T1w": 1, "T1gd": 2, "T2w": 3}


@dataclass
class SliceGeometry:
    """Where, in template space, the output slices are taken from.

    Attributes:
        z_indices: template z-indices (axial levels) that are kept.
        side: side length (in 1 mm template voxels) of the square in-plane crop.
        world_points: (side*side*n_slices, 3) template world coordinates of
            every voxel of the crop, in (x, y, z) 'ij' order.
        voxel_index: same points as template voxel indices (ints).
    """

    z_indices: np.ndarray
    side: int
    world_points: np.ndarray
    voxel_index: np.ndarray


def slice_geometry(template: Template, min_brain_fraction: float, stride: int) -> SliceGeometry:
    """Choose which axial slices to keep and the square crop around the brain."""
    mask = template.brain_mask

    # Axial levels: brain area per z, keep those with >= min_brain_fraction of the max.
    area = mask.sum(axis=(0, 1))
    z_keep = np.where(area >= min_brain_fraction * area.max())[0][::stride]

    # In-plane: bounding box of the brain over all z, made square, plus a margin.
    xs, ys = np.where(mask.any(axis=2))
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min())) + 8
    cx, cy = (xs.max() + xs.min()) // 2, (ys.max() + ys.min()) // 2
    x_range = np.arange(cx - side // 2, cx - side // 2 + side)
    y_range = np.arange(cy - side // 2, cy - side // 2 + side)

    grid = np.stack(np.meshgrid(x_range, y_range, z_keep, indexing="ij"), axis=-1).reshape(-1, 3)
    world = apply_affine(template.affine, grid.astype(np.float64))
    return SliceGeometry(z_indices=z_keep, side=side, world_points=world, voxel_index=grid)


def to_slices(values: np.ndarray, geom: SliceGeometry, size: int, binary: bool = False) -> np.ndarray:
    """Reshape sampled values to (n_slices, size, size) axial images.

    The 1 mm crop is shrunk to ``size`` x ``size`` with area averaging (a box
    filter), which avoids aliasing. Binary masks are thresholded at 0.5 after
    averaging. Images are rotated so anterior points up, as radiologists view them.
    """
    vol = values.reshape(geom.side, geom.side, len(geom.z_indices))      # (X, Y, Z)
    vol = np.rot90(vol, k=1, axes=(0, 1))                                # anterior up
    stack = torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 0, 1))).float()[:, None]
    small = F.interpolate(stack, size=(size, size), mode="area")[:, 0].numpy()
    return small > 0.5 if binary else small


def _robust_scale(volume: np.ndarray) -> np.ndarray:
    """Scale a raw scan so its bright tissue is ~1 (only used to help registration)."""
    positive = volume[volume > 0]
    p99 = np.percentile(positive, 99) if positive.size else 1.0
    return np.clip(volume / max(p99, 1e-6), 0.0, 1.5).astype(np.float32)


def load_canonical(path: str | Path, channel: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file reoriented to RAS. For 4D files pick ``channel``."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.dataobj[..., channel] if channel is not None else img.dataobj
    return np.asarray(data, dtype=np.float32), img.affine


def process_volume(
    image_path: str | Path,
    label_path: str | Path | None,
    cfg: dict,
    channel: int | None = None,
    already_skull_stripped: bool = False,
) -> dict:
    """Run the full pipeline on one scan and return its slices.

    Returns a dict with
        image: (n, S, S) float32 in [0, 1], zero outside the brain
        brain: (n, S, S) bool brain mask
        label: (n, S, S) bool tumour mask (all False when no label is given)
        z:     (n,) template z-index of each slice
        ncc:   registration quality (correlation with the template, 1 = perfect)
    """
    pp = cfg["preprocessing"]
    template = load_template()
    geom = slice_geometry(template, pp["min_brain_fraction"], pp["slice_stride"])
    size = pp["slice_size"]

    # 1. Load and roughly scale.
    volume, affine = load_canonical(image_path, channel)
    volume = _robust_scale(volume)

    # 2. Register to the template.
    reg = pp["registration"]
    matrix = register_affine(volume, affine, template, reg["levels_mm"], reg["iterations"])

    # Sample the scan on the template grid of the selected slices (1 mm).
    image_1mm = resample(volume, affine, matrix, geom.world_points)

    # 3. Skull-strip with the template brain mask (same for every subject).
    brain_1mm = template.brain_mask[tuple(geom.voxel_index.T)].astype(np.float32)
    if already_skull_stripped:
        # MSD/BraTS scans are zero outside the brain: also respect their own outline.
        own_brain = resample((volume > 0).astype(np.float32), affine, matrix, geom.world_points) > 0.5
        brain_1mm = brain_1mm * own_brain

    # Registration quality check: correlation with the template inside the brain.
    template_vals = template.image[tuple(geom.voxel_index.T)]
    inside = brain_1mm > 0
    quality = float(ncc(torch.tensor(template_vals[inside]), torch.tensor(image_1mm[inside])))

    # 4. Intensity normalisation inside the brain.
    p99 = np.percentile(image_1mm[inside], 99)
    image_1mm = np.clip(image_1mm / max(p99, 1e-6), 0.0, 1.0) * brain_1mm

    # 5. Slice and resize.
    image = to_slices(image_1mm, geom, size)
    brain = to_slices(brain_1mm, geom, size, binary=True)
    image = image * brain  # make sure nothing remains outside the (resized) mask

    if label_path is not None:
        label_vol, label_affine = load_canonical(label_path)
        tumour = (label_vol > 0).astype(np.float32)   # all tumour sub-regions = "whole tumour"
        label_1mm = resample(tumour, label_affine, matrix, geom.world_points)
        label = to_slices(label_1mm, geom, size, binary=True) & brain
    else:
        label = np.zeros_like(brain)

    return {
        "image": image.astype(np.float32),
        "brain": brain,
        "label": label,
        "z": geom.z_indices.copy(),
        "ncc": quality,
    }
