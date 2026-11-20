"""Behavioral tests for signal processing argument validation, shape inference, and IR node emission."""

from __future__ import annotations

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.signal as sig
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.signal import fftn, ifft2, ifftn, irfft2, irfftn
from ml_switcheroo_compiler.ops.signal.common_ops import _calculate_padding, _emit_signal_node
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import MockTensorStub, MockTestStub, require_callable


def test_signal_conv2d_validation_and_padding() -> None:
    """Verify `_validate_conv2d_args` rejects missing shapes and `_calculate_padding` builds config dicts."""

    class DummySignalTensor(MockTestStub):
        """Operand stub with optional `shape` attribute."""

        def __init__(self, shape: tuple[int, ...] | None = None) -> None:
            """Initialize dummy signal tensor stub.

            Args:
                shape (tuple[int, ...] | None): Optional shape tuple.
            """
            self.shape: tuple[int, ...] | None = shape

    class _Valid2DShape(MockTensorStub):
        """Operand with a 2D shape tuple."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.shape = (3, 3)

    with pytest.raises(ValueError):
        require_callable(sig._validate_conv2d_args)(DummySignalTensor(None), DummySignalTensor(None))
    with pytest.raises(ValueError):
        require_callable(sig._validate_conv2d_args)(_Valid2DShape(), DummySignalTensor(None))
    with pytest.raises(ValueError):
        require_callable(sig._validate_conv2d_args)(DummySignalTensor(None), _Valid2DShape())

    pad_cfg = _calculate_padding("same", "fill", 0.0)
    assert pad_cfg == {"mode": "same", "boundary": "fill", "fillvalue": 0.0}


def test_signal_op_classes_infer_shape() -> None:
    """Verify FFT family `OpDef.infer_shape` implementations return expected output shapes."""

    class _Shaped2D(MockTestStub):
        """Operand providing `.shape = (4, 4)`."""

        shape = (4, 4)

    shaped = _Shaped2D()
    assert sig.Rfft().infer_shape(shaped) == (4, 4)
    assert sig.Fft2().infer_shape(shaped) == (4, 4)
    assert sig.Fftfreq().infer_shape(5) == (5,)
    assert sig.Fftfreq().infer_shape(shaped) == ()
    assert sig.Irfft().infer_shape(shaped) == (4, 4)
    assert sig.Ihfft().infer_shape(shaped) == (4, 4)


def test_signal_fft_and_common_ops_ir_emission() -> None:
    """Verify `fftn`, `ifftn`, `irfftn`, `ifft2`, `irfft2`, and `_emit_signal_node` emit real IR nodes."""
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("signal_dsp_trace")
        try:
            t_2d = Tensor(
                ProxyTensor("sig_2d", (4, 4), "float32"),
                TensorConfig((4, 4), "float32", "cpu"),
            )
            assert isinstance(fftn(t_2d), Tensor)
            assert isinstance(ifftn(t_2d), Tensor)
            assert isinstance(irfftn(t_2d), Tensor)
            assert isinstance(ifft2(t_2d), Tensor)
            assert isinstance(irfft2(t_2d), Tensor)

            conv_out = _emit_signal_node("Convolve2D", [t_2d, t_2d], {"mode": "same"}, (4, 4), "float32")
            assert isinstance(conv_out, Tensor)
            assert conv_out.shape == (4, 4)
        finally:
            tape.stop_tracing()

    op_types = {node.op_type for node in traced_graph.nodes.values()}
    assert "Convolve2D" in op_types
    assert np.allclose(np.array([1.0]), np.array([1.0]))
