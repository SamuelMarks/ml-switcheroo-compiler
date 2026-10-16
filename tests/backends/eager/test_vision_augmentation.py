"""Tests for test_vision_augmentation."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
    """Test function test_flips."""
    img = np.array([[[1, 2], [3, 4]]])
    rng_high = random.Random()
    setattr(rng_high, "random", lambda: 0.6)
    res = cast("Callable[..., EagerTensorProtocol]", _flip_horizontal)(img, rng_high)
    assert id(res) != id(img)

    rng_low = random.Random()
    setattr(rng_low, "random", lambda: 0.4)
    res = cast("Callable[..., EagerTensorProtocol]", _flip_horizontal)(img, rng_low)
    assert id(res) == id(img)

    res = cast("Callable[..., EagerTensorProtocol]", _flip_vertical)(img, rng_high)
    assert id(res) != id(img)
    res = cast("Callable[..., EagerTensorProtocol]", _flip_vertical)(img, rng_low)
    assert id(res) == id(img)

    cast("Callable[..., EagerTensorProtocol]", _flip_both)(img, rng_high)
    cast("Callable[..., EagerTensorProtocol]", _flip_both)(img, rng_low)

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

    cfg = cast("Callable[..., EagerTensorProtocol]", RotationConfig)(factor=0.1, fill_mode="reflect", interpolation="bilinear", seed=42, fill_value=0.0, data_format=None)
    cast("Callable[..., EagerTensorProtocol]", _interpolate_pixels)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((1, 4, 4, 3)), cast("Callable[..., EagerTensorProtocol]", np.array)([[1]]), cast("Callable[..., EagerTensorProtocol]", np.array)([[1]]), cfg)

    cast("Callable[..., EagerTensorProtocol]", _compute_rotation_grid)(np, 4, 4, cast("Callable[..., EagerTensorProtocol]", random.Random)(), 0.1)

    cast("Callable[..., EagerTensorProtocol]", random_rotation_eager)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((1, 4, 4, 3)), cfg)
    assert cfg is not None


def test_crop_utils() -> None:
    """Test function test_crop_utils."""
    cast("Callable[..., EagerTensorProtocol]", _crop_and_pad_single)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((4, 4, 3)), cast("Callable[..., EagerTensorProtocol]", random.Random)(), (2, 2, 1, 1))
    _compute_random_crop(np, np.ones((1, 4, 4, 3)))
    random_crop_eager(np, np.ones((1, 4, 4, 3)), (2, 2))
    assert callable(random_crop_eager)


def test_perspective() -> None:
    """Test function test_perspective."""
    random_perspective_eager(np, np.ones((1, 4, 4, 3)), 0.1)
    assert callable(random_perspective_eager)


def test_elastic(mock_prepare: EagerTensorProtocol) -> None:
    """Test function test_elastic."""
    _blur_displacement(np, np.ones((1, 4, 4)), 0.1)
    cast("Callable[..., EagerTensorProtocol]", _generate_random_elastic_grid)(np, (1, 4, 4), cast("Callable[..., EagerTensorProtocol]", random.Random)(), 0.1, 0.1)

    assert isinstance(cast("Callable[..., EagerTensorProtocol]", _get_elastic_factor)(cast("Callable[..., EagerTensorProtocol]", random.Random)(), 0.1), float)
    assert isinstance(cast("Callable[..., EagerTensorProtocol]", _get_elastic_factor)(cast("Callable[..., EagerTensorProtocol]", random.Random)(), (0.1, 0.2)), float)

    with patch("ml_switcheroo_compiler.backends.eager.vision_augmentation._apply_elastic_batch", return_value=np.ones((1, 4, 4, 3))):
        random_elastic_transform_eager(np, np.ones((1, 4, 4, 3)), 0.1, 0.1)
        random_elastic_transform_eager(np, np.ones((1, 4, 4, 3)), (0.1, 0.2), (0.1, 0.2), interpolation="nearest")


def test_geometric(mock_prepare: EagerTensorProtocol) -> None:
    """Test function test_geometric."""
    rng = random.Random()
    cfg = cast("Callable[..., EagerTensorProtocol]", GeometricGridConfig)(H=4, W=4, rng=rng, factor1=0.1, factor2=0.1)
    cast("Callable[..., EagerTensorProtocol]", _compute_zoom_grid)(np, cfg)
    cast("Callable[..., EagerTensorProtocol]", _compute_translation_grid)(np, cfg)

    assert isinstance(cast("Callable[..., EagerTensorProtocol]", _get_shear_factor)(rng, 0.1), float)
    assert isinstance(cast("Callable[..., EagerTensorProtocol]", _get_shear_factor)(rng, (0.1, 0.2)), float)

    cast("Callable[..., EagerTensorProtocol]", _compute_shear_grid)(np, cfg)

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
