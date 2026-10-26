"""Affine image registration written in plain PyTorch.

Why not SimpleITK/ANTs? They have no Windows-on-ARM builds, and an affine
registration is small enough to write (and explain) ourselves:

* An affine transform maps a point ``w`` in template space (mm) to a point in
  the moving scan:  ``w_moving = c + B (w - c) + t``,
  where ``B`` (3x3) handles rotation/scaling/shearing, ``t`` is a translation
  and ``c`` is the centre of the template brain (so rotations happen around the
  brain, not the corner of the image).
* We look up the moving image at the transformed points with trilinear
  interpolation (``torch.nn.functional.grid_sample``), which is
  differentiable, so ``B`` and ``t`` can be fitted by gradient descent.
* The loss is the negative normalised cross-correlation (NCC) between the
  template and the resampled scan inside a region around the brain.
* We first score a grid of starting positions (tilt, shift, scale) at 6 mm,
  refine the best few briefly, then optimise coarse-to-fine (4 mm, then 2 mm).

Coordinate conventions used throughout:
    voxel index (i, j, k)  --affine-->  world coordinates in mm (x, y, z)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .template import Template


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def apply_affine(matrix: np.ndarray | torch.Tensor, points):
    """Apply a 4x4 homogeneous matrix to an (N, 3) array of points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def volume_to_tensor(volume: np.ndarray, device="cpu") -> torch.Tensor:
    """(I, J, K) numpy volume -> (1, 1, I, J, K) float tensor for grid_sample."""
    return torch.as_tensor(np.ascontiguousarray(volume), dtype=torch.float32, device=device)[None, None]


def sample_at_voxels(volume: torch.Tensor, voxels: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Interpolate ``volume`` (1,1,I,J,K) at fractional voxel coordinates (N,3).

    grid_sample wants coordinates normalised to [-1, 1] and ordered as
    (k, j, i) -- i.e. last array axis first -- so we convert here.
    Points outside the volume get the value 0.
    """
    sizes = torch.tensor(volume.shape[2:], dtype=voxels.dtype, device=voxels.device)
    normalised = 2.0 * voxels / (sizes - 1) - 1.0     # align_corners=True convention
    grid = normalised.flip(-1).view(1, 1, 1, -1, 3)    # (i,j,k) -> (k,j,i)
    # mode="bilinear" on a 5-D tensor is trilinear interpolation.
    out = F.grid_sample(volume, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return out.view(-1)


def ncc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalised cross-correlation of two 1-D tensors (1 = identical up to scale/offset)."""
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + eps)


def foreground_center(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """World-space centre of mass of the bright (non-background) voxels."""
    threshold = 0.1 * np.percentile(volume[volume > 0], 99) if np.any(volume > 0) else 0.0
    idx = np.argwhere(volume > threshold).astype(np.float64)
    return apply_affine(affine, idx.mean(axis=0))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _level_data(moving, voxel_sizes, template: Template, level_mm: int):
    """Points and intensities used by the loss at one resolution level.

    Fixed side: the template, blurred and sampled every ``level_mm`` voxels
    inside the registration ROI. Moving side: the scan blurred by a matching
    amount (sigma given per axis in voxels) so both are equally smooth.
    """
    fixed_smooth = ndimage.gaussian_filter(template.image, sigma=level_mm / 2)
    grid = np.zeros_like(template.roi_weight, dtype=bool)
    grid[::level_mm, ::level_mm, ::level_mm] = True
    pts_idx = np.argwhere(grid & (template.roi_weight > 0))
    fixed_vals = torch.tensor(fixed_smooth[tuple(pts_idx.T)], dtype=torch.float32)
    fixed_world = torch.tensor(apply_affine(template.affine, pts_idx.astype(np.float64)), dtype=torch.float32)
    moving_t = volume_to_tensor(ndimage.gaussian_filter(moving, sigma=(level_mm / 2) / voxel_sizes))
    return fixed_vals, fixed_world, moving_t


def register_affine(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    template: Template,
    levels_mm=(4, 2),
    iterations=(150, 100),
    lr_matrix: float = 2e-3,
    lr_translation: float = 0.5,
    max_points: int = 40_000,
    search_level_mm: int = 6,
    search_iterations: int = 60,
    search_pitch_deg=tuple(range(-25, 26, 5)),
    search_shift_y_mm=(-20, -10, 0, 10, 20),
    search_shift_z_mm=tuple(range(-40, 41, 10)),
    search_scale=(0.8, 0.9, 1.0),
    search_top_k: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Find the affine transform aligning ``moving`` to the template.

    Args:
        moving: (I, J, K) scan, roughly normalised to [0, 1].
        moving_affine: its 4x4 voxel->world matrix.
        template: the reference (see :mod:`brainad.template`).
        levels_mm: sampling resolution of each coarse-to-fine level.
        iterations: number of Adam steps per level.
        max_points: at most this many template points are used per step; a
            fresh random subset is drawn each step (stochastic optimisation,
            like ITK's "sampling percentage"). This makes the fine level ~5x
            faster with no loss of accuracy.
        search_*: grid of starting points (head tilt "pitch", shifts along the
            anterior-posterior (y) and inferior-superior (z) axes, overall scale).
            Gradient descent only finds the *nearest* good alignment, and for
            whole-head scans the centre-of-mass guess can be several cm off
            (how much neck is included varies) and heads can be tilted 20+
            degrees. Scoring a start costs a single interpolation, so we score
            every grid point, run a short coarse fit from the ``search_top_k``
            best, and continue from the winner.

    Returns:
        4x4 matrix ``M`` mapping template world coordinates to moving world
        coordinates (use it with :func:`resample`).
    """
    moving_affine = np.asarray(moving_affine, dtype=np.float64)
    inv_moving_affine = torch.tensor(np.linalg.inv(moving_affine), dtype=torch.float32)
    voxel_sizes = np.sqrt((moving_affine[:3, :3] ** 2).sum(axis=0))  # mm per voxel along each axis

    # Centre of rotation = centre of the template brain.
    brain_idx = np.argwhere(template.brain_mask).astype(np.float64)
    center = apply_affine(template.affine, brain_idx.mean(axis=0))
    center_t = torch.tensor(center, dtype=torch.float32)

    # Initial guess: shift so the centres of mass of the two images coincide.
    t0 = foreground_center(moving, moving_affine) - foreground_center(template.image, template.affine)
    t0_t = torch.tensor(t0, dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)

    def fit(A, t, level_data, n_iter) -> float:
        """Adam on the parameters B = I + A and translation t; returns the final NCC.

        Transform: template world -> moving world, w_m = c + B (w - c) + t + t0.
        """
        fixed_vals, fixed_world, moving_t = level_data
        optimizer = torch.optim.Adam([{"params": [A], "lr": lr_matrix}, {"params": [t], "lr": lr_translation}])
        for _ in range(n_iter):
            optimizer.zero_grad()
            # Random subset of template points for this step (all of them if few).
            if len(fixed_vals) > max_points:
                pick = torch.randperm(len(fixed_vals), generator=generator)[:max_points]
            else:
                pick = slice(None)
            moving_world = (fixed_world[pick] - center_t) @ (torch.eye(3) + A).T + center_t + t + t0_t
            moving_vals = sample_at_voxels(moving_t, apply_affine(inv_moving_affine, moving_world))
            loss = -ncc(fixed_vals[pick], moving_vals)
            loss.backward()
            optimizer.step()
        return -loss.item()

    # --- Stage 0: brute-force search for a good starting point ----------------
    search_data = _level_data(moving, voxel_sizes, template, search_level_mm)
    fixed_vals, fixed_world, moving_t = search_data
    candidates = []
    with torch.no_grad():
        for pitch in search_pitch_deg:
            c, s_ = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
            rot_x = torch.tensor([[1, 0, 0], [0, c, -s_], [0, s_, c]], dtype=torch.float32)  # nod forward/back
            for scale in search_scale:
                B = scale * rot_x
                rotated = (fixed_world - center_t) @ B.T + center_t + t0_t
                for dy in search_shift_y_mm:
                    for dz in search_shift_z_mm:
                        shift = torch.tensor([0.0, float(dy), float(dz)])
                        vals = sample_at_voxels(moving_t, apply_affine(inv_moving_affine, rotated + shift))
                        candidates.append((ncc(fixed_vals, vals).item(), B - torch.eye(3), shift))
    candidates.sort(key=lambda item: item[0], reverse=True)

    best = None
    for _, A0, t_init in candidates[:search_top_k]:
        A = A0.clone().requires_grad_(True)
        t = t_init.clone().requires_grad_(True)
        score = fit(A, t, search_data, search_iterations)
        if best is None or score > best[0]:
            best = (score, A.detach().clone(), t.detach().clone())
    A = best[1].requires_grad_(True)
    t = best[2].requires_grad_(True)

    # --- Stages 1..n: refine coarse-to-fine from the best start -------------
    for level_mm, n_iter in zip(levels_mm, iterations):
        fit(A, t, _level_data(moving, voxel_sizes, template, level_mm), n_iter)

    # Express the fitted transform as a single 4x4 matrix.
    with torch.no_grad():
        B = (torch.eye(3) + A).double().numpy()
        offset = center + t.double().numpy() + t0 - B @ center
    matrix = np.eye(4)
    matrix[:3, :3] = B
    matrix[:3, 3] = offset
    return matrix


def resample(
    volume: np.ndarray,
    volume_affine: np.ndarray,
    template_to_moving: np.ndarray,
    world_points: np.ndarray,
) -> np.ndarray:
    """Read ``volume`` at template-space ``world_points`` (N, 3) via the fitted transform.

    Uses trilinear interpolation; for binary masks, threshold the result at 0.5.
    """
    moving_world = apply_affine(template_to_moving, world_points.astype(np.float64))
    voxels = apply_affine(np.linalg.inv(volume_affine), moving_world)
    with torch.no_grad():
        values = sample_at_voxels(volume_to_tensor(volume), torch.tensor(voxels, dtype=torch.float32))
    return values.numpy()
