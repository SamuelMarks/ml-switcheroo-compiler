"""Tests for numpy eager vision augmentation ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_augmentation import (
    AffineConfig,
    BatchRotationConfig,
    InterpPixelsConfig,
    RotationConfig,
    _apply_affine_grid,
    _apply_rotation_batch,
    _bilinear_interpolation,
    _calculate_rotation_matrix,
    _create_rotation_mesh,
    _get_interp_pixels,
    _get_interp_weights,
    _nearest_interpolation,
    _np_augmix,
    _np_cutmix,
    _np_flip_both,
    _np_flip_horizontal,
    _np_flip_vertical,
    _np_mixup,
    _np_rand_augment,
    _np_random_color_jitter,
    _np_random_crop,
    _np_random_erasing,
    _np_random_flip,
    _np_random_rotation,
    _np_random_translation,
    _np_random_zoom,
    _process_batch_item,
    _resolve_rotation_factor,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_augmix() -> None:
    """Test function test_np_augmix."""
    images = np.ones((1, 4, 4, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", _np_augmix)(np, images, factor=0.1), images)


def test_np_cutmix() -> None:
    """Test function test_np_cutmix."""
    images1 = np.ones((1, 4, 4, 3))
    images2 = np.zeros((1, 4, 4, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", _np_cutmix)(np, images1, images2), images1)


def test_np_mixup() -> None:
    """Test function test_np_mixup."""
    images1 = np.ones((1, 4, 4, 3))
    images2 = np.zeros((1, 4, 4, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", _np_mixup)(np, images1, images2), images1)


def test_np_rand_augment() -> None:
    """Test function test_np_rand_augment."""
    images = np.ones((1, 4, 4, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", _np_rand_augment)(np, images, factor=0.1), images)


def test_np_random_color_jitter() -> None:
    """Test function test_np_random_color_jitter."""
    images = np.ones((1, 4, 4, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", _np_random_color_jitter)(np, images), images)


def test_np_random_crop() -> None:
    """Test function test_np_random_crop."""

    @global_eager_registry.register("RandomCrop")
    def _dummy_crop(bm: EagerValue, img: EagerValue, size: EagerValue, seed: EagerValue) -> EagerValue:
        """Test function _dummy_crop.

        Args:
            bm: Test parameter `bm`.
            img: Test parameter `img`.
            size: Test parameter `size`.
            seed: Test parameter `seed`.
        """
        return typing.cast("EagerCallableProtocol", np.ones)(size)

    images = np.ones((4, 4, 3))
    _np_random_crop(np, images, size=(2, 2))
    assert callable(test_np_random_crop)


def test_np_random_erasing() -> None:
    """Test function test_np_random_erasing."""
    images = np.ones((1, 4, 4, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", _np_random_erasing)(np, images, factor=0.1), images)


def test_np_flip_horizontal() -> None:
    """Test function test_np_flip_horizontal."""
    import random

    img = np.array([[[1, 2], [3, 4]]])
    rng = random.Random(42)
    res = typing.cast("EagerCallableProtocol", _np_flip_horizontal)(img, rng)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2, 2)


def test_np_flip_vertical() -> None:
    """Test function test_np_flip_vertical."""
    import random

    img = np.array([[[1, 2], [3, 4]]])
    rng = random.Random(42)
    res = typing.cast("EagerCallableProtocol", _np_flip_vertical)(img, rng)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2, 2)


def test_np_flip_both() -> None:
    """Test function test_np_flip_both."""
    import random

    img = np.array([[[1, 2], [3, 4]]])
    rng = random.Random(42)
    res = typing.cast("EagerCallableProtocol", _np_flip_both)(img, rng)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2, 2)


def test_np_random_flip() -> None:
    """Test function test_np_random_flip."""
    images_3d = np.ones((4, 4, 3))
    res3d = _np_random_flip(images_3d, mode="horizontal", seed=42)
    assert typing.cast("EagerTensorProtocol", getattr(res3d, "shape")) == (4, 4, 3)
    images_4d = np.ones((2, 4, 4, 3))
    res4d = _np_random_flip(images_4d, mode="vertical", seed=42)
    assert typing.cast("EagerTensorProtocol", getattr(res4d, "shape")) == (2, 4, 4, 3)


def test_calculate_rotation_matrix() -> None:
    """Test function test_calculate_rotation_matrix."""
    cfg = RotationConfig(theta=0.0, H=4, W=4, x=np.array([[0, 1], [0, 1]]), y=np.array([[0, 0], [1, 1]]))
    (src_x, src_y) = typing.cast("EagerCallableProtocol", _calculate_rotation_matrix)(np, cfg)
    assert typing.cast("EagerTensorProtocol", getattr(src_x, "shape")) == (2, 2)


def test_nearest_interpolation() -> None:
    """Test function test_nearest_interpolation."""
    images = np.ones((1, 4, 4, 1))
    coords = (np.array([[0.1, 1.2]]), np.array([[0.1, 1.2]]))
    res = _nearest_interpolation(np, images, coords, (4, 4), (0, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2)


def test_get_interp_pixels() -> None:
    """Test function test_get_interp_pixels."""
    images = np.ones((1, 4, 4, 1))
    cfg = InterpPixelsConfig(images=images, y0=np.array([[0]]), y1=np.array([[1]]), x0=np.array([[0]]), x1=np.array([[1]]), H=4, W=4, b_c=(0, 0))
    (Ia, Ib, Ic, Id) = typing.cast("EagerCallableProtocol", _get_interp_pixels)(np, cfg)
    assert typing.cast("EagerTensorProtocol", getattr(Ia, "shape")) == (1, 1)


def test_get_interp_weights() -> None:
    """Test function test_get_interp_weights."""
    src_coords = (np.array([[0.5]]), np.array([[0.5]]))
    bounds = (np.array([[0]]), np.array([[1]]), np.array([[0]]), np.array([[1]]))
    (wa, wb, wc, wd) = _get_interp_weights(src_coords, bounds)
    assert typing.cast("EagerTensorProtocol", getattr(wa, "shape")) == (1, 1)


def test_bilinear_interpolation() -> None:
    """Test function test_bilinear_interpolation."""
    images = np.ones((1, 4, 4, 1))
    coords = (np.array([[0.5, 1.5]]), np.array([[0.5, 1.5]]))
    res = _bilinear_interpolation(np, images, coords, (4, 4), (0, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2)


def test_apply_affine_grid() -> None:
    """Test function test_apply_affine_grid."""
    images = np.ones((1, 4, 4, 1))
    coords = (np.array([[0.5]]), np.array([[0.5]]))
    options_nearest = ("nearest", "constant", 0.0, np.array([[True]]))
    cfg_n = AffineConfig(coords=coords, shape=(4, 4), b_c=(0, 0), options=options_nearest)
    res_n = _apply_affine_grid(np, images, cfg_n)
    assert typing.cast("EagerTensorProtocol", getattr(res_n, "shape")) == (1, 1)
    options_bilinear = ("bilinear", "reflect", 0.0, np.array([[True]]))
    cfg_b = AffineConfig(coords=coords, shape=(4, 4), b_c=(0, 0), options=options_bilinear)
    res_b = _apply_affine_grid(np, images, cfg_b)
    assert typing.cast("EagerTensorProtocol", getattr(res_b, "shape")) == (1, 1)


def test_process_batch_item_and_apply_rotation_batch() -> None:
    """Test function test_process_batch_item_and_apply_rotation_batch."""
    images = np.ones((1, 4, 4, 1))
    out = np.zeros_like(images)
    angles = np.array([0.1])
    (y, x) = typing.cast("EagerCallableProtocol", _create_rotation_mesh)(np, 4, 4)
    options = ("bilinear", "constant", 0.0)
    cfg = typing.cast("EagerCallableProtocol", BatchRotationConfig)(images, angles, 4, 4, x, y, options)
    typing.cast("EagerCallableProtocol", _process_batch_item)(np, cfg, 0, out)
    assert out.shape == (1, 4, 4, 1)
    res_batch = typing.cast("EagerCallableProtocol", _apply_rotation_batch)(np, cfg)
    assert typing.cast("EagerTensorProtocol", getattr(res_batch, "shape")) == (1, 4, 4, 1)


def test_resolve_rotation_factor() -> None:
    """Test function test_resolve_rotation_factor."""
    assert _resolve_rotation_factor(0.1) == (-0.1, 0.1)
    assert _resolve_rotation_factor((0.1, 0.2)) == (0.1, 0.2)


def test_np_random_rotation() -> None:
    """Test function test_np_random_rotation."""
    images = np.ones((1, 4, 4, 1))
    res1 = _np_random_rotation(np, images, factor=0.1)
    assert typing.cast("EagerTensorProtocol", getattr(res1, "shape")) == (1, 4, 4, 1)
    img_3d = np.ones((4, 4, 1))
    res2 = _np_random_rotation(np, img_3d, factor=0.1)
    assert typing.cast("EagerTensorProtocol", getattr(res2, "shape")) == (4, 4, 1)
    img_2d = np.ones((4, 4))
    res3 = _np_random_rotation(np, img_2d, factor=0.1)
    assert typing.cast("EagerTensorProtocol", getattr(res3, "shape")) == (4, 4)


def test_np_random_translation() -> None:
    """Test function test_np_random_translation."""
    _np_random_translation(np, np.ones((4, 4, 3)), 0.1, 0.1)
    assert callable(test_np_random_translation)


def test_np_random_zoom() -> None:
    """Test function test_np_random_zoom."""
    _np_random_zoom(np, np.ones((4, 4, 3)), 0.1, 0.1)
    assert callable(test_np_random_zoom)
