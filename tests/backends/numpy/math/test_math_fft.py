"""Behavioral unit tests for math fft."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import math_fft as fft_module
from ml_switcheroo_compiler.backends.eager.reference_math.math_fft import (
    _np_fft,
    _np_fft2,
    _np_fftfreq,
    _np_ihfft,
    _np_irfft,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import (
    _np_fftn,
    _np_fftnd,
    _np_fftshift,
    _np_hfft,
    _np_ifft,
    _np_ifft2,
    _np_ifftn,
    _np_ifftnd,
    _np_ifftshift,
    _np_irfft2,
    _np_irfftn,
    _np_irfftnd,
    _np_rfft,
    _np_rfft2,
    _np_rfftfreq,
    _np_rfftn,
    _np_rfftnd,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import (
    EagerTensorProtocol,
    GenericNDArray,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_fft() -> None:
    """Test function test_math_fft."""
    a1d = np.ones((4,))
    a2d = np.ones((2, 2))
    a1d_complex = np.ones((4,), dtype=np.complex128)
    a2d_complex = np.ones((2, 2), dtype=np.complex128)

    assert extract_test_shape(_np_fft(np, a1d)) == (4,)
    assert extract_test_shape(_np_rfft(np, a1d)) == (3,)
    assert extract_test_shape(_np_ifft(np, a1d_complex)) == (4,)
    assert extract_test_shape(_np_irfft(np, a1d_complex)) == (6,)  # default n is 2*(m-1) => 2*(4-1) = 6

    assert extract_test_shape(_np_fftn(np, a2d)) == (2, 2)
    assert extract_test_shape(_np_ifftn(np, a2d_complex)) == (2, 2)
    assert extract_test_shape(_np_rfftn(np, a2d)) == (2, 2)
    assert extract_test_shape(_np_irfftn(np, a2d_complex)) == (2, 2)

    assert extract_test_shape(_np_fft2(np, a2d)) == (2, 2)
    assert extract_test_shape(_np_ifft2(np, a2d_complex)) == (2, 2)
    assert extract_test_shape(_np_rfft2(np, a2d)) == (2, 2)
    assert extract_test_shape(_np_irfft2(np, a2d_complex)) == (2, 2)

    assert extract_test_shape(_np_fftnd(np, a2d)) == (2, 2)
    assert extract_test_shape(_np_ifftnd(np, a2d_complex)) == (2, 2)
    assert extract_test_shape(_np_rfftnd(np, a2d)) == (2, 2)
    assert extract_test_shape(_np_irfftnd(np, a2d_complex)) == (2, 2)

    assert extract_test_shape(_np_fftshift(np, a1d)) == (4,)
    assert extract_test_shape(_np_ifftshift(np, a1d)) == (4,)

    assert extract_test_shape(_np_fftfreq(np, 4)) == (4,)
    assert extract_test_shape(_np_hfft(np, a1d)) == (6,)  # default n = 2*(m-1) = 6
    assert extract_test_shape(_np_ihfft(np, a1d)) == (3,)  # n=4 => m = n//2 + 1 = 3
    assert extract_test_shape(_np_rfftfreq(np, 4)) == (3,)


class _TestMathFftOpsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_fft_ops` split helpers."""

    c_sig2d: EagerTensorProtocol | OpValue
    r_sig2d: EagerTensorProtocol | OpValue
    signal_1d: GenericNDArray
    signal_2d: GenericNDArray


def _test_math_fft_ops_part_1(_state: _TestMathFftOpsState) -> None:
    """Execute part 1 of test_math_fft_ops.

    Args:
        _state: Shared test state dictionary.
    """
    signal_1d = np.array([1.0, 2.0, 1.0, -1.0])
    signal_2d = np.ones((4, 4), dtype=np.float64)

    assert len(coerce_operand_list(fft_module._np_fft(np, signal_1d))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Fft")(np, signal_1d)) == 4

    assert len(coerce_operand_list(fft_module._np_rfft(np, signal_1d))) == 3
    assert len(require_callable(numpy_eager_registry.get)("Rfft")(np, signal_1d)) == 3

    c_sig = fft_module._np_fft(np, signal_1d)
    assert len(coerce_operand_list(fft_module._np_ifft(np, c_sig))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Ifft")(np, c_sig)) == 4

    r_sig = fft_module._np_rfft(np, signal_1d)
    assert len(coerce_operand_list(fft_module._np_irfft(np, r_sig))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Irfft")(np, r_sig)) == 4

    assert extract_test_shape(fft_module._np_fftn(np, signal_2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Fftn")(np, signal_2d).shape == (4, 4)

    c_sig2d = fft_module._np_fftn(np, signal_2d)
    assert extract_test_shape(fft_module._np_ifftn(np, c_sig2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Ifftn")(np, c_sig2d).shape == (4, 4)

    assert extract_test_shape(fft_module._np_rfftn(np, signal_2d)) == (4, 3)
    assert require_callable(numpy_eager_registry.get)("Rfftn")(np, signal_2d).shape == (4, 3)

    r_sig2d = fft_module._np_rfftn(np, signal_2d)
    assert extract_test_shape(fft_module._np_irfftn(np, r_sig2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Irfftn")(np, r_sig2d).shape == (4, 4)
    _state.update({"c_sig2d": c_sig2d, "r_sig2d": r_sig2d})
    _state.update({"signal_1d": signal_1d, "signal_2d": signal_2d})


def _test_math_fft_ops_part_2(_state: _TestMathFftOpsState) -> None:
    """Execute part 2 of test_math_fft_ops.

    Args:
        _state: Shared test state dictionary.
    """
    c_sig2d, r_sig2d, signal_1d, signal_2d = (
        _state["c_sig2d"],
        _state["r_sig2d"],
        _state["signal_1d"],
        _state["signal_2d"],
    )
    assert extract_test_shape(fft_module._np_fft2(np, signal_2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Fft2")(np, signal_2d).shape == (4, 4)

    assert extract_test_shape(fft_module._np_ifft2(np, c_sig2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Ifft2")(np, c_sig2d).shape == (4, 4)

    assert extract_test_shape(fft_module._np_rfft2(np, signal_2d)) == (4, 3)
    assert require_callable(numpy_eager_registry.get)("Rfft2")(np, signal_2d).shape == (4, 3)

    assert extract_test_shape(fft_module._np_irfft2(np, r_sig2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Irfft2")(np, r_sig2d).shape == (4, 4)

    assert extract_test_shape(fft_module._np_fftnd(np, signal_2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Fftnd")(np, signal_2d).shape == (4, 4)

    assert extract_test_shape(fft_module._np_ifftnd(np, c_sig2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Ifftnd")(np, c_sig2d).shape == (4, 4)

    assert extract_test_shape(fft_module._np_rfftnd(np, signal_2d)) == (4, 3)
    assert require_callable(numpy_eager_registry.get)("Rfftnd")(np, signal_2d).shape == (4, 3)

    assert extract_test_shape(fft_module._np_irfftnd(np, r_sig2d)) == (4, 4)
    assert require_callable(numpy_eager_registry.get)("Irfftnd")(np, r_sig2d).shape == (4, 4)

    assert len(coerce_operand_list(fft_module._np_fftshift(np, signal_1d))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Fftshift")(np, signal_1d)) == 4

    assert len(coerce_operand_list(fft_module._np_ifftshift(np, signal_1d))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Ifftshift")(np, signal_1d)) == 4

    assert len(coerce_operand_list(fft_module._np_fftfreq(np, 4))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Fftfreq")(np, 4)) == 4
    _state.update({"c_sig2d": c_sig2d, "r_sig2d": r_sig2d, "signal_1d": signal_1d})
    _state.update({"signal_2d": signal_2d})


def _test_math_fft_ops_part_3(_state: _TestMathFftOpsState) -> None:
    """Execute part 3 of test_math_fft_ops.

    Args:
        _state: Shared test state dictionary.
    """
    signal_1d = _state["signal_1d"]
    assert len(coerce_operand_list(fft_module._np_hfft(np, np.array([1.0, 2.0, 1.0])))) == 4
    assert len(require_callable(numpy_eager_registry.get)("Hfft")(np, np.array([1.0, 2.0, 1.0]))) == 4

    assert len(coerce_operand_list(fft_module._np_ihfft(np, signal_1d))) == 3
    assert len(require_callable(numpy_eager_registry.get)("Ihfft")(np, signal_1d)) == 3

    assert len(coerce_operand_list(fft_module._np_rfftfreq(np, 4))) == 3
    assert len(require_callable(numpy_eager_registry.get)("Rfftfreq")(np, 4)) == 3


def test_math_fft_ops() -> None:
    """Test FFT operations directly and through eager registry.

    Returns:
        None
    """
    _state: _TestMathFftOpsState = {}
    _test_math_fft_ops_part_1(_state)
    _test_math_fft_ops_part_2(_state)
    _test_math_fft_ops_part_3(_state)
    assert len(_state) > 0
