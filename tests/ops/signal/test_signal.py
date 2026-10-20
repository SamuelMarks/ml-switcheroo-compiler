"""Tests for test_signal."""

from __future__ import annotations

import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_signal_missing() -> None:
    """Test function test_signal_missing."""
    import ml_switcheroo_compiler.ops.signal as sig

    class FakeShape:
        """Test class FakeShape."""

        shape = None

    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", sig._validate_conv2d_args)(typing.cast("EagerCallableProtocol", FakeShape)(), typing.cast("EagerCallableProtocol", FakeShape)())

    class FakeShape2:
        """Test class FakeShape2."""

        shape = (1, 1)

    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", sig._validate_conv2d_args)(typing.cast("EagerCallableProtocol", FakeShape2)(), typing.cast("EagerCallableProtocol", FakeShape)())

    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", sig._validate_conv2d_args)(typing.cast("EagerCallableProtocol", FakeShape)(), typing.cast("EagerCallableProtocol", FakeShape2)())


def test_signal_classes_infer_shape() -> None:
    """Test function test_signal_classes_infer_shape."""
    import ml_switcheroo_compiler.ops.signal as sig

    class DummyData:
        """Test class DummyData."""

        shape = (1, 1)

    assert typing.cast("EagerCallableProtocol", sig.Rfft)().infer_shape(typing.cast("EagerCallableProtocol", DummyData)()) == (1, 1)
    assert typing.cast("EagerCallableProtocol", sig.Fft2)().infer_shape(typing.cast("EagerCallableProtocol", DummyData)()) == (1, 1)
    assert typing.cast("EagerCallableProtocol", sig.Fftfreq)().infer_shape(5) == (5,)
    assert typing.cast("EagerCallableProtocol", sig.Fftfreq)().infer_shape(typing.cast("EagerCallableProtocol", DummyData)()) == ()
    assert typing.cast("EagerCallableProtocol", sig.Irfft)().infer_shape(typing.cast("EagerCallableProtocol", DummyData)()) == (1, 1)
    assert typing.cast("EagerCallableProtocol", sig.Ihfft)().infer_shape(typing.cast("EagerCallableProtocol", DummyData)()) == (1, 1)


def test_signal_real_dsp_validation() -> None:
    """Test function test_signal_real_dsp_validation."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.signal import fftn, ifft2, ifftn, irfft2, irfftn

    Tensor(np.array([1.0, 2.0, 3.0, 4.0]), TensorConfig((4,), "float32", "cpu"))
    t_2d = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            assert isinstance(fftn(t_2d), Tensor)
            assert isinstance(ifftn(t_2d), Tensor)
            assert isinstance(irfftn(t_2d), Tensor)
            assert isinstance(ifft2(t_2d), Tensor)
            assert isinstance(irfft2(t_2d), Tensor)
