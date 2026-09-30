"""Tests for test_math_fft_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_fft as fft_module


def test_math_fft_ops() -> None:
    """Test FFT operations directly and through eager registry.

    Returns:
        None
    """
    signal_1d = np.array([1.0, 2.0, 1.0, -1.0])
    signal_2d = np.ones((4, 4), dtype=np.float64)

    assert len(fft_module._np_fft(np, signal_1d)) == 4
    assert len(numpy_eager_registry.get("Fft")(np, signal_1d)) == 4

    assert len(fft_module._np_rfft(np, signal_1d)) == 3
    assert len(numpy_eager_registry.get("Rfft")(np, signal_1d)) == 3

    c_sig = fft_module._np_fft(np, signal_1d)
    assert len(fft_module._np_ifft(np, c_sig)) == 4
    assert len(numpy_eager_registry.get("Ifft")(np, c_sig)) == 4

    r_sig = fft_module._np_rfft(np, signal_1d)
    assert len(fft_module._np_irfft(np, r_sig)) == 4
    assert len(numpy_eager_registry.get("Irfft")(np, r_sig)) == 4

    assert fft_module._np_fftn(np, signal_2d).shape == (4, 4)
    assert numpy_eager_registry.get("Fftn")(np, signal_2d).shape == (4, 4)

    c_sig2d = fft_module._np_fftn(np, signal_2d)
    assert fft_module._np_ifftn(np, c_sig2d).shape == (4, 4)
    assert numpy_eager_registry.get("Ifftn")(np, c_sig2d).shape == (4, 4)

    assert fft_module._np_rfftn(np, signal_2d).shape == (4, 3)
    assert numpy_eager_registry.get("Rfftn")(np, signal_2d).shape == (4, 3)

    r_sig2d = fft_module._np_rfftn(np, signal_2d)
    assert fft_module._np_irfftn(np, r_sig2d).shape == (4, 4)
    assert numpy_eager_registry.get("Irfftn")(np, r_sig2d).shape == (4, 4)

    assert fft_module._np_fft2(np, signal_2d).shape == (4, 4)
    assert numpy_eager_registry.get("Fft2")(np, signal_2d).shape == (4, 4)

    assert fft_module._np_ifft2(np, c_sig2d).shape == (4, 4)
    assert numpy_eager_registry.get("Ifft2")(np, c_sig2d).shape == (4, 4)

    assert fft_module._np_rfft2(np, signal_2d).shape == (4, 3)
    assert numpy_eager_registry.get("Rfft2")(np, signal_2d).shape == (4, 3)

    assert fft_module._np_irfft2(np, r_sig2d).shape == (4, 4)
    assert numpy_eager_registry.get("Irfft2")(np, r_sig2d).shape == (4, 4)

    assert fft_module._np_fftnd(np, signal_2d).shape == (4, 4)
    assert numpy_eager_registry.get("Fftnd")(np, signal_2d).shape == (4, 4)

    assert fft_module._np_ifftnd(np, c_sig2d).shape == (4, 4)
    assert numpy_eager_registry.get("Ifftnd")(np, c_sig2d).shape == (4, 4)

    assert fft_module._np_rfftnd(np, signal_2d).shape == (4, 3)
    assert numpy_eager_registry.get("Rfftnd")(np, signal_2d).shape == (4, 3)

    assert fft_module._np_irfftnd(np, r_sig2d).shape == (4, 4)
    assert numpy_eager_registry.get("Irfftnd")(np, r_sig2d).shape == (4, 4)

    assert len(fft_module._np_fftshift(np, signal_1d)) == 4
    assert len(numpy_eager_registry.get("Fftshift")(np, signal_1d)) == 4

    assert len(fft_module._np_ifftshift(np, signal_1d)) == 4
    assert len(numpy_eager_registry.get("Ifftshift")(np, signal_1d)) == 4

    assert len(fft_module._np_fftfreq(np, 4)) == 4
    assert len(numpy_eager_registry.get("Fftfreq")(np, 4)) == 4

    assert len(fft_module._np_hfft(np, np.array([1.0, 2.0, 1.0]))) == 4
    assert len(numpy_eager_registry.get("Hfft")(np, np.array([1.0, 2.0, 1.0]))) == 4

    assert len(fft_module._np_ihfft(np, signal_1d)) == 3
    assert len(numpy_eager_registry.get("Ihfft")(np, signal_1d)) == 3

    assert len(fft_module._np_rfftfreq(np, 4)) == 3
    assert len(numpy_eager_registry.get("Rfftfreq")(np, 4)) == 3
