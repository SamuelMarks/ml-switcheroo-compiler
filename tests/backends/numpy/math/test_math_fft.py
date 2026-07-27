"""Tests for numpy eager math fft ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_fft import (
    _np_fft,
    _np_fft2,
    _np_fftfreq,
    _np_fftn,
    _np_fftnd,
    _np_fftshift,
    _np_hfft,
    _np_ifft,
    _np_ifft2,
    _np_ifftn,
    _np_ifftnd,
    _np_ifftshift,
    _np_ihfft,
    _np_irfft,
    _np_irfft2,
    _np_irfftn,
    _np_irfftnd,
    _np_rfft,
    _np_rfft2,
    _np_rfftfreq,
    _np_rfftn,
    _np_rfftnd,
)


def test_math_fft() -> None:
    a1d = np.ones((4,))
    a2d = np.ones((2, 2))
    a1d_complex = np.ones((4,), dtype=np.complex128)
    a2d_complex = np.ones((2, 2), dtype=np.complex128)

    assert _np_fft(np, a1d).shape == (4,)
    assert _np_rfft(np, a1d).shape == (3,)
    assert _np_ifft(np, a1d_complex).shape == (4,)
    assert _np_irfft(np, a1d_complex).shape == (6,)  # default n is 2*(m-1) => 2*(4-1) = 6

    assert _np_fftn(np, a2d).shape == (2, 2)
    assert _np_ifftn(np, a2d_complex).shape == (2, 2)
    assert _np_rfftn(np, a2d).shape == (2, 2)
    assert _np_irfftn(np, a2d_complex).shape == (2, 2)

    assert _np_fft2(np, a2d).shape == (2, 2)
    assert _np_ifft2(np, a2d_complex).shape == (2, 2)
    assert _np_rfft2(np, a2d).shape == (2, 2)
    assert _np_irfft2(np, a2d_complex).shape == (2, 2)

    assert _np_fftnd(np, a2d).shape == (2, 2)
    assert _np_ifftnd(np, a2d_complex).shape == (2, 2)
    assert _np_rfftnd(np, a2d).shape == (2, 2)
    assert _np_irfftnd(np, a2d_complex).shape == (2, 2)

    assert _np_fftshift(np, a1d).shape == (4,)
    assert _np_ifftshift(np, a1d).shape == (4,)

    assert _np_fftfreq(np, 4).shape == (4,)
    assert _np_hfft(np, a1d).shape == (6,)  # default n = 2*(m-1) = 6
    assert _np_ihfft(np, a1d).shape == (3,)  # n=4 => m = n//2 + 1 = 3
    assert _np_rfftfreq(np, 4).shape == (3,)
