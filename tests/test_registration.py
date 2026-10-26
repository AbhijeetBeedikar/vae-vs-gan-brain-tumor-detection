"""The registration must recover a known affine transform."""

import numpy as np

from brainad.registration import apply_affine, register_affine, resample
from brainad.template import load_template


def test_recovers_known_affine():
    template = load_template()

    # Ground truth: rotate 8 degrees, scale by 1.08, shift by a few mm.
    theta = np.deg2rad(8)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    true = np.eye(4)
    true[:3, :3] = 1.08 * rot
    true[:3, 3] = [6, -9, 4]

    # Build a "moving" scan on a different 1.5 mm grid by warping the template.
    moving_affine = np.diag([1.5, 1.5, 1.5, 1.0])
    moving_affine[:3, 3] = [-120, -150, -100]
    shape = (160, 190, 150)
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    moving = resample(template.image, template.affine, np.linalg.inv(true), apply_affine(moving_affine, idx))
    moving = moving.reshape(shape)

    fitted = register_affine(moving, moving_affine, template)

    pts = apply_affine(template.affine, np.argwhere(template.brain_mask)[::200].astype(float))
    error_mm = np.linalg.norm(apply_affine(fitted, pts) - apply_affine(true, pts), axis=1).mean()
    assert error_mm < 1.0, f"mean registration error {error_mm:.2f} mm"
