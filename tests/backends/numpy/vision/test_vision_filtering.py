"""Unit tests for NumPy eager vision spatial filtering and patch extraction operations."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.vision_ops import filtering as vision_filtering_module
from ml_switcheroo_compiler.backends.eager.vision_ops.filtering import (
    _np_degeneration,
    _np_gaussian_blur,
    _np_median_filter,
    _np_sharpen,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.scipy_ndimage_kernels import _np_extract_volume_patches
from ml_switcheroo_compiler.ops.configs import BlurConfig
from tests.testing_utils import require_callable


def test_vision_filtering_exports_and_identity_ops() -> None:
    """Verify module `__all__` exports alongside `_np_degeneration` and `_np_sharpen`."""
    exported = dir(vision_filtering_module)
    for symbol in vision_filtering_module.__all__:
        assert symbol in exported

    mod = np
    imgs = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)

    np.testing.assert_array_equal(_np_degeneration(mod, imgs), imgs)
    reg_degen = numpy_eager_registry.get("Degeneration")
    assert reg_degen is not None
    np.testing.assert_array_equal(np.asarray(reg_degen(np, imgs)), imgs)

    np.testing.assert_array_equal(_np_sharpen(mod, imgs), imgs)
    reg_sharp = numpy_eager_registry.get("Sharpen")
    assert reg_sharp is not None
    np.testing.assert_array_equal(np.asarray(reg_sharp(np, imgs)), imgs)


def test_np_gaussian_blur_real_execution_and_rank_check() -> None:
    """Verify `_np_gaussian_blur` applies normalized 2D Gaussian convolution for kwargs and `BlurConfig`."""
    mod = np
    with pytest.raises(ValueError, match="Images must be 4D"):
        _np_gaussian_blur(mod, np.ones((2, 2), dtype=np.float32))

    const_img = np.full((1, 1, 4, 4), 2.5, dtype=np.float32)
    cfg = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    blurred_const = _np_gaussian_blur(mod, const_img, config=cfg)
    assert blurred_const.shape == (1, 1, 4, 4)
    np.testing.assert_allclose(blurred_const, const_img, atol=1e-5)

    impulse = np.zeros((1, 1, 5, 5), dtype=np.float32)
    impulse[0, 0, 2, 2] = 1.0

    blur_kw = _np_gaussian_blur(mod, impulse, kernel_size=(3, 3), sigma=(1.0, 1.0))
    blur_cfg = _np_gaussian_blur(mod, impulse, config=cfg)
    np.testing.assert_allclose(blur_kw, blur_cfg, atol=1e-6)
    assert np.isclose(float(np.sum(blur_kw)), 1.0, atol=1e-5)
    assert float(blur_kw[0, 0, 2, 2]) > float(blur_kw[0, 0, 2, 1]) > float(blur_kw[0, 0, 1, 1]) > 0.0
    assert float(blur_kw[0, 0, 0, 0]) == 0.0

    reg_blur = numpy_eager_registry.get("GaussianBlur")
    assert reg_blur is not None
    reg_out = np.asarray(reg_blur(np, impulse, config=cfg))
    np.testing.assert_allclose(reg_out, blur_cfg, atol=1e-6)


def test_np_median_filter_real_execution_and_missing_kernel_size() -> None:
    """Verify `_np_median_filter` removes single-pixel impulse spikes on 4D images."""
    mod = np
    with pytest.raises(ValueError, match="Images must be 4D"):
        _np_median_filter(mod, np.ones((2, 2), dtype=np.float32))

    noisy = np.ones((1, 1, 5, 5), dtype=np.float32)
    noisy[0, 0, 2, 2] = 100.0

    med_reflect = np.asarray(require_callable(_np_median_filter)(np, noisy, kernel_size=(3, 3), padding="reflect"))
    assert med_reflect.shape == (1, 1, 5, 5)
    np.testing.assert_allclose(med_reflect, np.ones((1, 1, 5, 5), dtype=np.float32))

    reg_med = numpy_eager_registry.get("MedianFilter")
    assert reg_med is not None
    med_same = np.asarray(reg_med(np, noisy, kernel_size=(3, 3), padding="same"))
    assert med_same.shape == (1, 1, 5, 5)
    np.testing.assert_allclose(med_same[0, 0, 1:4, 1:4], np.ones((3, 3), dtype=np.float32))
    assert float(med_same[0, 0, 0, 0]) == 0.0


def test_np_extract_volume_patches_branches() -> None:
    """Verify `_np_extract_volume_patches` across empty, non-5D, valid 5D, and oversized-window inputs."""
    mod = np
    assert _np_extract_volume_patches(mod).size == 0

    mat_2d = np.ones((2, 2), dtype=np.float32)
    np.testing.assert_array_equal(_np_extract_volume_patches(mod, mat_2d), mat_2d)

    vol = np.arange(32, dtype=np.float32).reshape(1, 2, 2, 4, 2)
    ksizes = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 2, 1]
    patches = _np_extract_volume_patches(mod, vol, ksizes, strides)
    assert patches.shape == (1, 1, 1, 2, 2, 8)
    np.testing.assert_allclose(patches[0, 0, 0, 0, 0], vol[0, :, :, 0:2, 0].ravel())

    oversized = [1, 5, 5, 5, 1]
    unit_strides = [1, 1, 1, 1, 1]
    fallback_vol = _np_extract_volume_patches(mod, vol, oversized, unit_strides)
    np.testing.assert_array_equal(fallback_vol, vol)
