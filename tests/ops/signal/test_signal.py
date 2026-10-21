"""Tests for test_signal."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_signal_missing() -> None:
    """Test function test_signal_missing."""
    import ml_switcheroo_compiler.ops.signal as sig

    class FakeShape:
        """Test class FakeShape."""

        shape = None

    with pytest.raises(ValueError):
        as_callable(sig._validate_conv2d_args)(as_callable(FakeShape)(), as_callable(FakeShape)())

    class FakeShape2:
        """Test class FakeShape2."""

        shape = (1, 1)

    with pytest.raises(ValueError):
        as_callable(sig._validate_conv2d_args)(as_callable(FakeShape2)(), as_callable(FakeShape)())

    with pytest.raises(ValueError):
        as_callable(sig._validate_conv2d_args)(as_callable(FakeShape)(), as_callable(FakeShape2)())


def test_signal_classes_infer_shape() -> None:
    """Test function test_signal_classes_infer_shape."""
    import ml_switcheroo_compiler.ops.signal as sig

    class DummyData:
        """Test class DummyData."""

        shape = (1, 1)

    assert as_callable(sig.Rfft)().infer_shape(as_callable(DummyData)()) == (1, 1)
    assert as_callable(sig.Fft2)().infer_shape(as_callable(DummyData)()) == (1, 1)
    assert as_callable(sig.Fftfreq)().infer_shape(5) == (5,)
    assert as_callable(sig.Fftfreq)().infer_shape(as_callable(DummyData)()) == ()
    assert as_callable(sig.Irfft)().infer_shape(as_callable(DummyData)()) == (1, 1)
    assert as_callable(sig.Ihfft)().infer_shape(as_callable(DummyData)()) == (1, 1)


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
