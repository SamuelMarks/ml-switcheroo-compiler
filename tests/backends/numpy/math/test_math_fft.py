"""Tests for numpy eager math fft ops."""

from __future__ import annotations

import typing

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
from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_math_fft() -> None:
    """Test function test_math_fft."""
    a1d = np.ones((4,))
    a2d = np.ones((2, 2))
    a1d_complex = np.ones((4,), dtype=np.complex128)
    a2d_complex = np.ones((2, 2), dtype=np.complex128)

    assert ensure_eager_tensor(_np_fft(ensure_eager_tensor(np), a1d)).shape == (4,)
    assert ensure_eager_tensor(_np_rfft(ensure_eager_tensor(np), a1d)).shape == (3,)
    assert ensure_eager_tensor(_np_ifft(ensure_eager_tensor(np), a1d_complex)).shape == (4,)
    assert ensure_eager_tensor(_np_irfft(ensure_eager_tensor(np), a1d_complex)).shape == (6,)  # default n is 2*(m-1) => 2*(4-1) = 6

    assert ensure_eager_tensor(_np_fftn(ensure_eager_tensor(np), a2d)).shape == (2, 2)
    assert ensure_eager_tensor(_np_ifftn(ensure_eager_tensor(np), a2d_complex)).shape == (2, 2)
    assert ensure_eager_tensor(_np_rfftn(ensure_eager_tensor(np), a2d)).shape == (2, 2)
    assert ensure_eager_tensor(_np_irfftn(ensure_eager_tensor(np), a2d_complex)).shape == (2, 2)

    assert ensure_eager_tensor(_np_fft2(ensure_eager_tensor(np), a2d)).shape == (2, 2)
    assert ensure_eager_tensor(_np_ifft2(ensure_eager_tensor(np), a2d_complex)).shape == (2, 2)
    assert ensure_eager_tensor(_np_rfft2(ensure_eager_tensor(np), a2d)).shape == (2, 2)
    assert ensure_eager_tensor(_np_irfft2(ensure_eager_tensor(np), a2d_complex)).shape == (2, 2)

    assert ensure_eager_tensor(_np_fftnd(ensure_eager_tensor(np), a2d)).shape == (2, 2)
    assert ensure_eager_tensor(_np_ifftnd(ensure_eager_tensor(np), a2d_complex)).shape == (2, 2)
    assert ensure_eager_tensor(_np_rfftnd(ensure_eager_tensor(np), a2d)).shape == (2, 2)
    assert ensure_eager_tensor(_np_irfftnd(ensure_eager_tensor(np), a2d_complex)).shape == (2, 2)

    assert ensure_eager_tensor(_np_fftshift(ensure_eager_tensor(np), a1d)).shape == (4,)
    assert ensure_eager_tensor(_np_ifftshift(ensure_eager_tensor(np), a1d)).shape == (4,)

    assert ensure_eager_tensor(_np_fftfreq(ensure_eager_tensor(np), 4)).shape == (4,)
    assert ensure_eager_tensor(_np_hfft(ensure_eager_tensor(np), a1d)).shape == (6,)  # default n = 2*(m-1) = 6
    assert ensure_eager_tensor(_np_ihfft(ensure_eager_tensor(np), a1d)).shape == (3,)  # n=4 => m = n//2 + 1 = 3
    assert ensure_eager_tensor(_np_rfftfreq(ensure_eager_tensor(np), 4)).shape == (3,)
