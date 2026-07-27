"""Test module."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.signal import _apply_conv2d_batch, _apply_median_filter_batch, _generate_gaussian_kernel, _get_blur_config, gaussian_blur_eager, median_filter_eager


class DummyConfig:
    pass


def test_signal():
    import numpy as np_mod

    # 1. _generate_gaussian_kernel
    kernel = _generate_gaussian_kernel(np_mod, (3, 3), (1.0, 1.0))
    assert kernel.shape == (3, 3)
    np.testing.assert_allclose(np_mod.sum(kernel), 1.0)

    # 2. _apply_conv2d_batch
    imgs = np.ones((1, 1, 3, 3))
    out = _apply_conv2d_batch(np_mod, imgs, kernel, "reflect")
    assert out.shape == (1, 1, 3, 3)

    with pytest.raises(ValueError):
        _apply_conv2d_batch(np_mod, np.ones((3, 3)), kernel, "reflect")

    # 3. _get_blur_config
    c1 = _get_blur_config({"kernel_size": (5, 5)}, None)
    assert c1.kernel_size == (5, 5)
    c2 = DummyConfig()
    assert _get_blur_config({}, c2) is c2

    # 4. gaussian_blur_eager
    imgs2 = np.ones((1, 1, 3, 3))
    res1 = gaussian_blur_eager(np_mod, imgs2, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert res1.shape == (1, 1, 3, 3)

    # 5. _apply_median_filter_batch
    imgs_med = np.random.rand(1, 1, 3, 3)
    out_med1 = _apply_median_filter_batch(np_mod, imgs_med, (3, 3), "same")
    assert out_med1.shape == (1, 1, 3, 3)

    out_med2 = _apply_median_filter_batch(np_mod, imgs_med, (3, 3), "reflect")
    assert out_med2.shape == (1, 1, 3, 3)

    with pytest.raises(ValueError):
        _apply_median_filter_batch(np_mod, np.ones((3, 3)), (3, 3), "same")

    # 6. median_filter_eager
    res2 = median_filter_eager(np_mod, imgs_med, (3, 3))
    assert res2.shape == (1, 1, 3, 3)
