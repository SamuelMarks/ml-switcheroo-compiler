import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    AffineTransformParams,
    RotationConfig,
    _apply_affine_transform,
    _blur_displacement,
    _compute_random_crop,
    _compute_rotation_grid,
    _compute_rotation_matrix,
    _compute_shear_grid,
    _compute_translation_grid,
    _compute_zoom_grid,
    _crop_and_pad_single,
    _flip_both,
    _flip_horizontal,
    _flip_vertical,
    _generate_coordinate_grid,
    _generate_random_elastic_grid,
    _get_elastic_factor,
    _get_shear_factor,
    _interpolate_pixels,
    random_crop_eager,
    random_elastic_transform_eager,
    random_flip_eager,
    random_perspective_eager,
    random_rotation_eager,
    random_shear_eager,
    random_translation_eager,
    random_zoom_eager,
)
from ml_switcheroo_compiler.backends.eager.vision_utils import GeometricGridConfig


@pytest.fixture
def mock_prepare():
    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._prepare_eager_transform") as m:
        ctx = MagicMock()
        ctx.np_mod = np
        ctx.imgs = np.ones((1, 4, 4, 3))
        ctx.B = 1
        ctx.H = 4
        ctx.W = 4
        ctx.rng = random.Random(42)
        m.return_value = ctx
        yield m


def test_flips(mock_prepare):
    img = np.array([[[1, 2], [3, 4]]])
    rng_high = random.Random()
    rng_high.random = lambda: 0.6
    res = _flip_horizontal(img, rng_high)
    assert res is not img

    rng_low = random.Random()
    rng_low.random = lambda: 0.4
    res = _flip_horizontal(img, rng_low)
    assert res is img

    res = _flip_vertical(img, rng_high)
    assert res is not img
    res = _flip_vertical(img, rng_low)
    assert res is img

    _flip_both(img, rng_high)
    _flip_both(img, rng_low)

    imgs = np.ones((1, 4, 4, 3))
    random_flip_eager(np, imgs, mode="horizontal", seed=42)
    random_flip_eager(np, imgs, mode="vertical", seed=42)
    random_flip_eager(np, imgs, mode="horizontal_and_vertical", seed=42)
    random_flip_eager(np, imgs, mode="invalid", seed=42)


def test_rotation_utils():
    _compute_rotation_matrix(np, 0.1, 4, 4)
    _generate_coordinate_grid(np, 4, 4)
    params = AffineTransformParams(cos_a=1.0, sin_a=0.0, cx=2.0, cy=2.0)
    _apply_affine_transform(np.array([[1]]), np.array([[1]]), params)

    cfg = RotationConfig(factor=0.1, fill_mode="reflect", interpolation="bilinear", seed=42, fill_value=0.0, data_format=None)
    _interpolate_pixels(np, np.ones((1, 4, 4, 3)), np.array([[1]]), np.array([[1]]), cfg)

    _compute_rotation_grid(np, 4, 4, random.Random(), 0.1)

    random_rotation_eager(np, np.ones((1, 4, 4, 3)), cfg)


def test_crop_utils():
    _crop_and_pad_single(np, np.ones((4, 4, 3)), random.Random(), (2, 2, 1, 1))
    _compute_random_crop(np, np.ones((1, 4, 4, 3)))
    random_crop_eager(np, np.ones((1, 4, 4, 3)), (2, 2))


def test_perspective():
    random_perspective_eager(np, np.ones((1, 4, 4, 3)), 0.1)


def test_elastic(mock_prepare):
    _blur_displacement(np, np.ones((1, 4, 4)), 0.1)
    _generate_random_elastic_grid(np, (1, 4, 4), random.Random(), 0.1, 0.1)

    assert isinstance(_get_elastic_factor(random.Random(), 0.1), float)
    assert isinstance(_get_elastic_factor(random.Random(), (0.1, 0.2)), float)

    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._apply_elastic_batch", return_value=np.ones((1, 4, 4, 3))):
        random_elastic_transform_eager(np, np.ones((1, 4, 4, 3)), 0.1, 0.1)
        random_elastic_transform_eager(np, np.ones((1, 4, 4, 3)), (0.1, 0.2), (0.1, 0.2), interpolation="nearest")


def test_geometric(mock_prepare):
    rng = random.Random()
    cfg = GeometricGridConfig(H=4, W=4, rng=rng, factor1=0.1, factor2=0.1)
    _compute_zoom_grid(np, cfg)
    _compute_translation_grid(np, cfg)

    assert isinstance(_get_shear_factor(rng, 0.1), float)
    assert isinstance(_get_shear_factor(rng, (0.1, 0.2)), float)

    _compute_shear_grid(np, cfg)

    imgs = np.ones((1, 4, 4, 3))
    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._compute_zoom_grid", return_value=(0, 0)):
        random_zoom_eager(np, imgs, height_factor=0.1)
        random_zoom_eager(np, imgs, height_factor=(0.1, 0.2), width_factor=None)
        random_zoom_eager(np, imgs, height_factor=0.1, width_factor=0.2)

    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._compute_translation_grid", return_value=(0, 0)):
        random_translation_eager(np, imgs, height_factor=0.1, width_factor=0.1)

    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._compute_shear_grid", return_value=(0, 0)):
        random_shear_eager(np, imgs, y_factor=0.1)
        random_shear_eager(np, imgs, y_factor=(0.1, 0.2), x_factor=None)
