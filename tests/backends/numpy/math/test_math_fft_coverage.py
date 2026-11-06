"""Tests for test_math_fft_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_fft as fft_module
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_math_fft_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_fft_ops.

    Args:
        _state: Shared test state dictionary.
    """
    signal_1d = np.array([1.0, 2.0, 1.0, -1.0])
    signal_2d = np.ones((4, 4), dtype=np.float64)

    assert ensure_callable(len)(ensure_callable(fft_module._np_fft)(np, signal_1d)) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Fft")(np, signal_1d)) == 4

    assert ensure_callable(len)(ensure_callable(fft_module._np_rfft)(np, signal_1d)) == 3
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Rfft")(np, signal_1d)) == 3

    c_sig = fft_module._np_fft(ensure_eager_tensor(np), signal_1d)
    assert ensure_callable(len)(ensure_callable(fft_module._np_ifft)(np, c_sig)) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Ifft")(np, c_sig)) == 4

    r_sig = fft_module._np_rfft(ensure_eager_tensor(np), signal_1d)
    assert ensure_callable(len)(ensure_callable(fft_module._np_irfft)(np, r_sig)) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Irfft")(np, r_sig)) == 4

    assert ensure_eager_tensor(fft_module._np_fftn(ensure_eager_tensor(np), signal_2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Fftn")(np, signal_2d)).shape == (4, 4)

    c_sig2d = fft_module._np_fftn(ensure_eager_tensor(np), signal_2d)
    assert ensure_eager_tensor(fft_module._np_ifftn(ensure_eager_tensor(np), c_sig2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Ifftn")(np, c_sig2d)).shape == (4, 4)

    assert ensure_eager_tensor(fft_module._np_rfftn(ensure_eager_tensor(np), signal_2d)).shape == (4, 3)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Rfftn")(np, signal_2d)).shape == (4, 3)

    r_sig2d = fft_module._np_rfftn(ensure_eager_tensor(np), signal_2d)
    assert ensure_eager_tensor(fft_module._np_irfftn(ensure_eager_tensor(np), r_sig2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Irfftn")(np, r_sig2d)).shape == (4, 4)
    _state.update(ensure_tensor_dict({"c_sig": c_sig, "c_sig2d": c_sig2d, "r_sig": r_sig, "r_sig2d": r_sig2d}))
    _state.update(ensure_tensor_dict({"signal_1d": signal_1d, "signal_2d": signal_2d}))


def _test_math_fft_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_fft_ops.

    Args:
        _state: Shared test state dictionary.
    """
    c_sig2d, r_sig2d, signal_1d, signal_2d = _state["c_sig2d"], _state["r_sig2d"], _state["signal_1d"], _state["signal_2d"]
    assert ensure_eager_tensor(fft_module._np_fft2(ensure_eager_tensor(np), signal_2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Fft2")(np, signal_2d)).shape == (4, 4)

    assert ensure_eager_tensor(fft_module._np_ifft2(ensure_eager_tensor(np), c_sig2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Ifft2")(np, c_sig2d)).shape == (4, 4)

    assert ensure_eager_tensor(fft_module._np_rfft2(ensure_eager_tensor(np), signal_2d)).shape == (4, 3)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Rfft2")(np, signal_2d)).shape == (4, 3)

    assert ensure_eager_tensor(fft_module._np_irfft2(ensure_eager_tensor(np), r_sig2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Irfft2")(np, r_sig2d)).shape == (4, 4)

    assert ensure_eager_tensor(fft_module._np_fftnd(ensure_eager_tensor(np), signal_2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Fftnd")(np, signal_2d)).shape == (4, 4)

    assert ensure_eager_tensor(fft_module._np_ifftnd(ensure_eager_tensor(np), c_sig2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Ifftnd")(np, c_sig2d)).shape == (4, 4)

    assert ensure_eager_tensor(fft_module._np_rfftnd(ensure_eager_tensor(np), signal_2d)).shape == (4, 3)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Rfftnd")(np, signal_2d)).shape == (4, 3)

    assert ensure_eager_tensor(fft_module._np_irfftnd(ensure_eager_tensor(np), r_sig2d)).shape == (4, 4)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Irfftnd")(np, r_sig2d)).shape == (4, 4)

    assert ensure_callable(len)(ensure_callable(fft_module._np_fftshift)(np, signal_1d)) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Fftshift")(np, signal_1d)) == 4

    assert ensure_callable(len)(ensure_callable(fft_module._np_ifftshift)(np, signal_1d)) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Ifftshift")(np, signal_1d)) == 4

    assert ensure_callable(len)(ensure_callable(fft_module._np_fftfreq)(np, 4)) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Fftfreq")(np, 4)) == 4
    _state.update(ensure_tensor_dict({"c_sig2d": c_sig2d, "r_sig2d": r_sig2d, "signal_1d": signal_1d}))
    _state.update(ensure_tensor_dict({"signal_2d": signal_2d}))


def _test_math_fft_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_fft_ops.

    Args:
        _state: Shared test state dictionary.
    """
    signal_1d = _state["signal_1d"]
    assert ensure_callable(len)(ensure_callable(fft_module._np_hfft)(np, ensure_callable(np.array)([1.0, 2.0, 1.0]))) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Hfft")(np, ensure_callable(np.array)([1.0, 2.0, 1.0]))) == 4

    assert ensure_callable(len)(ensure_callable(fft_module._np_ihfft)(np, signal_1d)) == 3
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Ihfft")(np, signal_1d)) == 3

    assert ensure_callable(len)(ensure_callable(fft_module._np_rfftfreq)(np, 4)) == 3
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Rfftfreq")(np, 4)) == 3


def test_math_fft_ops() -> None:
    """Test FFT operations directly and through eager registry.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_fft_ops_part_1(_state)
    _test_math_fft_ops_part_2(_state)
    _test_math_fft_ops_part_3(_state)
    assert len(_state) > 0
