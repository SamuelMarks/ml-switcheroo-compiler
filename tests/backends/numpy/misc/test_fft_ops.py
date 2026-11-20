"""Tests for numpy eager fft ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.reference_math.fft_ops import (
    FFTConfig,
    _np_fft3d,
    _np_ifft3d,
    _np_irfft2d,
    _np_irfft3d,
    _np_istft,
    _np_rfft2d,
    _np_rfft3d,
    _np_stft,
    _np_window_hamming,
    _np_window_hann,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import (
    _np_fftnd,
    _np_ifftnd,
    _np_irfftnd,
    _np_rfftnd,
)
from tests.testing_utils import extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_fft3d() -> None:
    """Test fft3d."""
    a = np.ones((2, 2, 2))
    res = _np_fft3d(np, a)
    assert extract_test_shape(res) == (2, 2, 2)
    # config branch
    res_cfg = _np_fft3d(np, a, config=FFTConfig())
    assert extract_test_shape(res_cfg) == (2, 2, 2)


def test_ifft3d() -> None:
    """Test ifft3d."""
    a = np.ones((2, 2, 2), dtype=np.complex128)
    res = _np_ifft3d(np, a)
    assert extract_test_shape(res) == (2, 2, 2)


def test_rfft2d() -> None:
    """Test rfft2d."""
    a = np.ones((2, 2))
    res = _np_rfft2d(np, a)
    assert extract_test_shape(res) == (2, 2)
    res_cfg = _np_rfft2d(np, a, config=FFTConfig())
    assert extract_test_shape(res_cfg) == (2, 2)


def test_rfft3d() -> None:
    """Test rfft3d."""
    a = np.ones((2, 2, 2))
    res = _np_rfft3d(np, a)
    assert extract_test_shape(res) == (2, 2, 2)


def test_irfft2d() -> None:
    """Test irfft2d."""
    a = np.ones((2, 2), dtype=np.complex128)
    res = _np_irfft2d(np, a)
    assert extract_test_shape(res) == (2, 2)


def test_irfft3d() -> None:
    """Test irfft3d."""
    a = np.ones((2, 2, 2), dtype=np.complex128)
    res = _np_irfft3d(np, a)
    assert extract_test_shape(res) == (2, 2, 2)


def test_fftnd() -> None:
    """Test fftnd."""
    a = np.ones((2, 2))
    res = _np_fftnd(np, a)
    assert extract_test_shape(res) == (2, 2)


def test_ifftnd() -> None:
    """Test ifftnd."""
    a = np.ones((2, 2), dtype=np.complex128)
    res = _np_ifftnd(np, a)
    assert extract_test_shape(res) == (2, 2)


def test_rfftnd() -> None:
    """Test rfftnd."""
    a = np.ones((2, 2))
    res = _np_rfftnd(np, a)
    assert extract_test_shape(res) == (2, 2)


def test_irfftnd() -> None:
    """Test irfftnd."""
    a = np.ones((2, 2), dtype=np.complex128)
    res = _np_irfftnd(np, a)
    assert extract_test_shape(res) == (2, 2)


def test_window_hann() -> None:
    """Test window_hann."""
    res = _np_window_hann(np, 10)
    assert extract_test_shape(res) == (10,)


def test_window_hamming() -> None:
    """Test window_hamming."""
    res = _np_window_hamming(np, 10)
    assert extract_test_shape(res) == (10,)


def test_stft() -> None:
    """Test stft."""
    x = np.ones((10,))
    res = _np_stft(np, x, 4, 2)
    assert extract_test_shape(res) == (3, 4)

    with pytest.raises(ValueError, match="noverlap must be less than nfft"):
        _np_stft(np, x, 4, 4)


def test_istft() -> None:
    """Test istft."""
    x = np.ones((3, 4), dtype=np.complex128)
    res = _np_istft(np, x, 4, 2)
    assert extract_test_shape(res) == (10,)

    with pytest.raises(ValueError, match="noverlap must be less than nfft"):
        _np_istft(np, x, 4, 4)
