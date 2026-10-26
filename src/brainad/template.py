"""A T2-like brain template in MNI152 space.

Every scan (healthy IXI and tumour MSD) is aligned to one common template so
that anatomy lines up across subjects and across the two datasets.

nilearn ships the ICBM 2009a MNI152 template at 1 mm, but only as a T1 image
plus grey-matter (GM) and white-matter (WM) probability maps. Registering a
T2 scan to a T1 image is hard because the contrasts are inverted (CSF is dark
on T1 but bright on T2). Instead we *synthesise* a T2-like template from the
tissue maps, using the typical relative brightness of each tissue on T2:

    CSF (fluid)  -> bright   (1.00)
    grey matter  -> medium   (0.55)
    white matter -> darker   (0.35)

Same-contrast images can then be aligned with a simple correlation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

# Relative T2 brightness of each tissue class (see module docstring).
T2_INTENSITY = {"csf": 1.0, "gm": 0.55, "wm": 0.35}


@dataclass
class Template:
    """Everything we need to know about the reference space.

    Attributes:
        image:  (X, Y, Z) synthetic T2 template, values in [0, 1].
        brain_mask: (X, Y, Z) bool, the brain (incl. ventricles) in template space.
        roi_weight: (X, Y, Z) float, region used by the registration loss:
            the brain plus a margin, so the loss also sees the brain outline.
        affine: 4x4 voxel-index -> world (mm, RAS) matrix of the template.
    """

    image: np.ndarray
    brain_mask: np.ndarray
    roi_weight: np.ndarray
    affine: np.ndarray


@lru_cache(maxsize=1)
def load_template() -> Template:
    """Build (once per process) the synthetic T2 MNI template at 1 mm."""
    # Imported lazily so modules that don't need the template don't pay for nilearn.
    from nilearn.datasets import load_mni152_gm_template, load_mni152_wm_template

    gm_img = load_mni152_gm_template(resolution=1)
    wm_img = load_mni152_wm_template(resolution=1)
    gm = gm_img.get_fdata().astype(np.float32)
    wm = wm_img.get_fdata().astype(np.float32)
    affine = gm_img.affine.copy()

    # Brain mask = voxels that are mostly tissue, with holes (the ventricles)
    # filled in, and small gaps closed.
    brain = (gm + wm) > 0.2
    brain = ndimage.binary_closing(brain, iterations=2)
    brain = ndimage.binary_fill_holes(brain)

    # CSF probability = whatever is neither GM nor WM, restricted to the inside
    # of the head. We include a 3 mm shell around the brain because real T2
    # scans show bright CSF in the sulci and around the brain surface.
    head = ndimage.binary_dilation(brain, iterations=3)
    csf = np.clip(1.0 - gm - wm, 0.0, 1.0) * head

    image = (
        T2_INTENSITY["csf"] * csf
        + T2_INTENSITY["gm"] * gm
        + T2_INTENSITY["wm"] * wm
    ).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    # Registration region: brain + ~8 mm margin. Using only this region keeps
    # the neck and scalp of whole-head IXI scans from dominating the loss.
    roi = ndimage.binary_dilation(brain, iterations=8).astype(np.float32)

    return Template(image=image, brain_mask=brain, roi_weight=roi, affine=affine)
