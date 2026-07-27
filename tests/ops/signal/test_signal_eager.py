"""Test dummy ops and signal functions."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.signal import (
    _apply_conv2d_batch,
    _apply_median_filter_batch,
    _generate_gaussian_kernel,
    gaussian_blur_eager,
    median_filter_eager,
)


def test_signal_eager() -> None:
    """Test signal ops eager mode."""
    kernel = _generate_gaussian_kernel(np, (3, 3), (1.0, 1.0))
    assert kernel.shape == (3, 3)

    imgs = np.ones((1, 1, 4, 4))
    res = _apply_conv2d_batch(np, imgs, kernel, "reflect")
    assert res.shape == (1, 1, 4, 4)

    # 3d error
    with pytest.raises(ValueError):
        _apply_conv2d_batch(np, np.ones((4, 4)), kernel, "reflect")

    res = gaussian_blur_eager(np, imgs, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert res.shape == (1, 1, 4, 4)

    res = _apply_median_filter_batch(np, imgs, (3, 3), "same")
    assert res.shape == (1, 1, 4, 4)

    # 3d error
    with pytest.raises(ValueError):
        _apply_median_filter_batch(np, np.ones((4, 4)), (3, 3), "same")

    res = median_filter_eager(np, imgs, (3, 3), "same")
    assert res.shape == (1, 1, 4, 4)
