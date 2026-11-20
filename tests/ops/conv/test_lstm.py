"""Tests for test_lstm."""

from __future__ import annotations

import importlib
import types
import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.conv_lstm as mod
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_lstm import conv_lstm_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import RNNWeights
from tests.testing_utils import MockTestStub, coerce_int_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
ConvLSTMConfig = importlib.import_module("ml_switcheroo_compiler.ops.nn.conv_lstm").ConvLSTMConfig


def test_conv_lstm_cov() -> None:
    """Test function test_conv_lstm_cov."""
    orig_eager = config.eager_mode
    orig_backend = config.backend
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackend(BaseGenerator):
            """Test class MockBackend."""

            @classmethod
            def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function execute_op.

                Args:
                    op_type (str): Test parameter `op_type`.
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `execute_op`.
                """
                del op_type, args, kwargs
                return np.ones((2, 2, 2, 2))

            @classmethod
            def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
                """Test function array.

                Args:
                    data (BackendArg): Test parameter `data`.
                    dtype (BackendArg | None): Test parameter `dtype`.

                Returns:
                    BackendArg: Computed output value from `array`.
                """
                del dtype
                return data

        BackendRegistry.register("mock_ops", MockBackend)
        tc3 = TensorConfig(
            shape=(2, 2, 2),
            dtype=types.SimpleNamespace(value="float32"),
            device=None,
            requires_grad=False,
            trainable=False,
        )
        tc4 = TensorConfig(
            shape=(2, 2, 2, 2),
            dtype=types.SimpleNamespace(value="float32"),
            device=None,
            requires_grad=False,
            trainable=False,
        )
        tc5 = TensorConfig(
            shape=(2, 2, 2, 2, 2),
            dtype=types.SimpleNamespace(value="float32"),
            device=None,
            requires_grad=False,
            trainable=False,
        )

        class DummyTensor(MockTestStub):
            """Test class DummyTensor."""

            def __init__(self, shape: BackendArg) -> None:
                """Test function __init__.

                Args:
                    shape (BackendArg): Test parameter `shape`.
                """
                self.shape = shape
                self.data = np.ones(coerce_int_list(shape))
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t3 = Tensor(DummyTensor((2, 2, 2)), tc3)
        t4 = Tensor(DummyTensor((2, 2, 2, 2)), tc4)
        t5 = Tensor(DummyTensor((2, 2, 2, 2, 2)), tc5)
        w = RNNWeights(kernel=t4, recurrent_kernel=t4, bias=t4)
        c = ConvLSTMConfig((3, 3))
        with pytest.raises(ValueError):
            conv_lstm_cell(t3, (t3, t3), w, c)
        with pytest.raises(ValueError):
            conv_lstm_cell(t4, (t4, t4), w, c)
        with pytest.raises(ValueError):
            conv_lstm_cell(t5, (t5, t5), w, c)
        with pytest.raises(ValueError):
            conv_lstm_cell(Tensor(DummyTensor2D(), tc3), (t3, t3), w, c)
        with pytest.raises(ValueError):
            conv_lstm_cell(
                Tensor(
                    DummyTensor2D(),
                    TensorConfig(
                        shape=(2, 2),
                        dtype=types.SimpleNamespace(value="float32"),
                        device=None,
                        requires_grad=False,
                        trainable=False,
                    ),
                ),
                (t3, t3),
                w,
                c,
            )
    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend


def test_conv_lstm_gates() -> None:
    """Test function test_conv_lstm_gates."""
    from ml_switcheroo_compiler.ops.nn.conv_lstm import _apply_conv_lstm_gates

    orig_eager = config.eager_mode
    orig_backend = config.backend
    try:
        config.eager_mode = True
        config.backend = "mock_ops"
        tc4 = TensorConfig(
            shape=(2, 2, 2, 2),
            dtype=types.SimpleNamespace(value="float32"),
            device=None,
            requires_grad=False,
            trainable=False,
        )

        class DummyTensor(MockTestStub):
            """Test class DummyTensor."""

            def __init__(self, shape: BackendArg) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape = shape
                self.data = np.ones(coerce_int_list(shape))
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t4 = Tensor(DummyTensor((2, 2, 2, 2)), tc4)
        w = RNNWeights(kernel=t4, recurrent_kernel=t4)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod, "split", lambda *args, **kwargs: (t4, t4, t4, t4), raising=False)
            mp.setattr(mod, "_sigmoid", lambda x: x)
            mp.setattr(mod, "tanh", lambda x: x, raising=False)
            mp.setattr(mod, "add", lambda *args, **kwargs: t4, raising=False)
            mp.setattr(mod, "multiply", lambda *args, **kwargs: t4, raising=False)
            mod._apply_conv_lstm_gates(t4, t4, (t4, t4), w, "channels_last")
            w2 = RNNWeights(kernel=t4, recurrent_kernel=t4, bias=t4)
            mod._apply_conv_lstm_gates(t4, t4, (t4, t4), w2, "channels_last")
        with pytest.raises(ValueError):
            _apply_conv_lstm_gates(t4, t4, (t4, t4), w, "channels_last")
        with pytest.raises(ValueError):
            _apply_conv_lstm_gates(t4, t4, (t4, t4), w, "channels_first")
    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend


class DummyTensor2D:
    """Test class DummyTensor2D."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.shape = (2, 2)
        self.data = np.ones((2, 2))
        self.dtype = types.SimpleNamespace(value="float32")
        self.device = None


def test_lstm_sigmoid() -> None:
    """Test function test_lstm_sigmoid."""
    from ml_switcheroo_compiler.ops.nn.conv_lstm import _sigmoid
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    orig_eager = config.eager_mode
    global_tracing_state.start_tracing()
    try:
        config.eager_mode = False
        out = _sigmoid(
            Tensor(
                DummyTensor2D(),
                TensorConfig(
                    shape=(2, 2),
                    dtype=types.SimpleNamespace(value="float32"),
                    device=None,
                    requires_grad=False,
                    trainable=False,
                ),
            )
        )
        assert global_tracing_state.active_graph is not None and str(out) != ""
        assert any(n.op_type == "Sigmoid" for n in global_tracing_state.active_graph.nodes.values())
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = orig_eager
