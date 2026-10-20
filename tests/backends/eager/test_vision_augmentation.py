"""Tests for test_vision_augmentation."""

from __future__ import annotations

import random
import typing
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

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


@pytest.fixture
def mock_prepare() -> Iterator[EagerValue]:
    """Test function mock_prepare."""
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


def test_flips(mock_prepare: EagerTensorProtocol) -> None:
    """Test function test_flips.

    Args:
        mock_prepare: Test parameter `mock_prepare`.
    """
    img = np.array([[[1, 2], [3, 4]]])
    rng_high = random.Random()
    setattr(rng_high, "random", lambda: 0.6)
    res = typing.cast("EagerCallableProtocol", _flip_horizontal)(img, rng_high)
    assert id(res) != id(img)

    rng_low = random.Random()
    setattr(rng_low, "random", lambda: 0.4)
    res = typing.cast("EagerCallableProtocol", _flip_horizontal)(img, rng_low)
    assert id(res) == id(img)

    res = typing.cast("EagerCallableProtocol", _flip_vertical)(img, rng_high)
    assert id(res) != id(img)
    res = typing.cast("EagerCallableProtocol", _flip_vertical)(img, rng_low)
    assert id(res) == id(img)

    typing.cast("EagerCallableProtocol", _flip_both)(img, rng_high)
    typing.cast("EagerCallableProtocol", _flip_both)(img, rng_low)

    imgs = np.ones((1, 4, 4, 3))
    random_flip_eager(np, imgs, mode="horizontal", seed=42)
    random_flip_eager(np, imgs, mode="vertical", seed=42)
    random_flip_eager(np, imgs, mode="horizontal_and_vertical", seed=42)
    random_flip_eager(np, imgs, mode="invalid", seed=42)


def test_rotation_utils() -> None:
    """Test function test_rotation_utils."""
    _compute_rotation_matrix(np, 0.1, 4, 4)
    _generate_coordinate_grid(np, 4, 4)
    params = AffineTransformParams(cos_a=1.0, sin_a=0.0, cx=2.0, cy=2.0)
    _apply_affine_transform(np.array([[1]]), np.array([[1]]), params)

    cfg = typing.cast("EagerCallableProtocol", RotationConfig)(factor=0.1, fill_mode="reflect", interpolation="bilinear", seed=42, fill_value=0.0, data_format=None)
    typing.cast("EagerCallableProtocol", _interpolate_pixels)(np, typing.cast("EagerCallableProtocol", np.ones)((1, 4, 4, 3)), typing.cast("EagerCallableProtocol", np.array)([[1]]), typing.cast("EagerCallableProtocol", np.array)([[1]]), cfg)

    typing.cast("EagerCallableProtocol", _compute_rotation_grid)(np, 4, 4, typing.cast("EagerCallableProtocol", random.Random)(), 0.1)

    typing.cast("EagerCallableProtocol", random_rotation_eager)(np, typing.cast("EagerCallableProtocol", np.ones)((1, 4, 4, 3)), cfg)
    assert cfg is not None


def test_crop_utils() -> None:
    """Test function test_crop_utils."""
    typing.cast("EagerCallableProtocol", _crop_and_pad_single)(np, typing.cast("EagerCallableProtocol", np.ones)((4, 4, 3)), typing.cast("EagerCallableProtocol", random.Random)(), (2, 2, 1, 1))
    _compute_random_crop(np, np.ones((1, 4, 4, 3)))
    random_crop_eager(np, np.ones((1, 4, 4, 3)), (2, 2))
    assert callable(random_crop_eager)


def test_perspective() -> None:
    """Test function test_perspective."""
    random_perspective_eager(np, np.ones((1, 4, 4, 3)), 0.1)
    assert callable(random_perspective_eager)


def test_elastic(mock_prepare: EagerTensorProtocol) -> None:
    """Test function test_elastic.

    Args:
        mock_prepare: Test parameter `mock_prepare`.
    """
    _blur_displacement(np, np.ones((1, 4, 4)), 0.1)
    typing.cast("EagerCallableProtocol", _generate_random_elastic_grid)(np, (1, 4, 4), typing.cast("EagerCallableProtocol", random.Random)(), 0.1, 0.1)

    assert isinstance(typing.cast("EagerCallableProtocol", _get_elastic_factor)(typing.cast("EagerCallableProtocol", random.Random)(), 0.1), float)
    assert isinstance(typing.cast("EagerCallableProtocol", _get_elastic_factor)(typing.cast("EagerCallableProtocol", random.Random)(), (0.1, 0.2)), float)

    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._apply_elastic_batch", return_value=np.ones((1, 4, 4, 3))):
        random_elastic_transform_eager(np, np.ones((1, 4, 4, 3)), 0.1, 0.1)
        random_elastic_transform_eager(np, np.ones((1, 4, 4, 3)), (0.1, 0.2), (0.1, 0.2), interpolation="nearest")


def test_geometric(mock_prepare: EagerTensorProtocol) -> None:
    """Test function test_geometric.

    Args:
        mock_prepare: Test parameter `mock_prepare`.
    """
    rng = random.Random()
    cfg = typing.cast("EagerCallableProtocol", GeometricGridConfig)(H=4, W=4, rng=rng, factor1=0.1, factor2=0.1)
    typing.cast("EagerCallableProtocol", _compute_zoom_grid)(np, cfg)
    typing.cast("EagerCallableProtocol", _compute_translation_grid)(np, cfg)

    assert isinstance(typing.cast("EagerCallableProtocol", _get_shear_factor)(rng, 0.1), float)
    assert isinstance(typing.cast("EagerCallableProtocol", _get_shear_factor)(rng, (0.1, 0.2)), float)

    typing.cast("EagerCallableProtocol", _compute_shear_grid)(np, cfg)

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
