"""Tests for test_lstm."""

from __future__ import annotations

import importlib
import types
import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.conv_lstm as mod
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_lstm import conv_lstm_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import RNNWeights
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue
ConvLSTMConfig = ensure_eager_tensor(ensure_eager_tensor(importlib.import_module("ml_switcheroo_compiler.ops.nn.conv_lstm")).ConvLSTMConfig)


def test_conv_lstm_cov() -> None:
    """Test function test_conv_lstm_cov."""
    orig_eager = config.eager_mode
    orig_backend = config.backend
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackend:
            """Test class MockBackend."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op.

                Args:
                    op_name: Test parameter `op_name`.
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    EagerValue: Computed output value from `execute_op`.
                """
                return np.ones((2, 2, 2, 2))

            @classmethod
            def array(cls, data: EagerValue) -> EagerValue:
                """Test function array.

                Args:
                    data: Test parameter `data`.

                Returns:
                    EagerValue: Computed output value from `array`.
                """
                return data

        ensure_callable(BackendRegistry.register)("mock_ops", MockBackend)
        tc3 = ensure_callable(TensorConfig)(shape=(2, 2, 2), dtype=ensure_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)
        tc4 = ensure_callable(TensorConfig)(shape=(2, 2, 2, 2), dtype=ensure_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)
        tc5 = ensure_callable(TensorConfig)(shape=(2, 2, 2, 2, 2), dtype=ensure_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self, shape: EagerValue) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape = shape
                self.data = ensure_callable(np.ones)(shape)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t3 = ensure_callable(Tensor)(ensure_callable(DummyTensor)((2, 2, 2)), tc3)
        t4 = ensure_callable(Tensor)(ensure_callable(DummyTensor)((2, 2, 2, 2)), tc4)
        t5 = ensure_callable(Tensor)(ensure_callable(DummyTensor)((2, 2, 2, 2, 2)), tc5)
        w = ensure_callable(RNNWeights)(t4, t4, t4)
        c = ensure_callable(ConvLSTMConfig)((3, 3))
        with pytest.raises(ValueError):
            ensure_callable(conv_lstm_cell)(t3, (t3, t3), w, c)
        with pytest.raises(ValueError):
            ensure_callable(conv_lstm_cell)(t4, (t4, t4), w, c)
        with pytest.raises(ValueError):
            ensure_callable(conv_lstm_cell)(t5, (t5, t5), w, c)
        with pytest.raises(ValueError):
            ensure_callable(conv_lstm_cell)(ensure_callable(Tensor)(ensure_callable(DummyTensor2D)(), tc3), (t3, t3), w, c)
        with pytest.raises(ValueError):
            ensure_callable(conv_lstm_cell)(
                ensure_callable(Tensor)(ensure_callable(DummyTensor2D)(), ensure_callable(TensorConfig)(shape=(2, 2), dtype=ensure_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)),
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
        tc4 = ensure_callable(TensorConfig)(shape=(2, 2, 2, 2), dtype=ensure_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self, shape: EagerValue) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape = shape
                self.data = ensure_callable(np.ones)(shape)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t4 = ensure_callable(Tensor)(ensure_callable(DummyTensor)((2, 2, 2, 2)), tc4)
        w = ensure_callable(RNNWeights)(t4, t4)
        orig_split = ensure_eager_tensor(ensure_eager_tensor(mod).split)
        orig_sigmoid = mod._sigmoid
        orig_tanh = ensure_eager_tensor(ensure_eager_tensor(mod).tanh)
        orig_add = ensure_eager_tensor(ensure_eager_tensor(mod).add)
        orig_multiply = ensure_eager_tensor(ensure_eager_tensor(mod).multiply)
        try:
            ensure_mutable(mod).split = ensure_eager_tensor(lambda *args, **kwargs: (t4, t4, t4, t4))
            mod._sigmoid = lambda x: x
            ensure_mutable(mod).tanh = ensure_eager_tensor(lambda x: x)
            ensure_mutable(mod).add = ensure_eager_tensor(lambda *args, **kwargs: t4)
            ensure_mutable(mod).multiply = ensure_eager_tensor(lambda *args, **kwargs: t4)
            ensure_callable(mod._apply_conv_lstm_gates)(t4, t4, (t4, t4), w, "channels_last")
            w2 = ensure_callable(RNNWeights)(t4, t4, bias=t4)
            ensure_callable(mod._apply_conv_lstm_gates)(t4, t4, (t4, t4), w2, "channels_last")
        finally:
            ensure_mutable(mod).split = ensure_eager_tensor(orig_split)
            mod._sigmoid = orig_sigmoid
            ensure_mutable(mod).tanh = ensure_eager_tensor(orig_tanh)
            ensure_mutable(mod).add = ensure_eager_tensor(orig_add)
            ensure_mutable(mod).multiply = ensure_eager_tensor(orig_multiply)
        with pytest.raises(ValueError):
            ensure_callable(_apply_conv_lstm_gates)(t4, t4, (t4, t4), w, "channels_last")
        with pytest.raises(ValueError):
            ensure_callable(_apply_conv_lstm_gates)(t4, t4, (t4, t4), w, "channels_first")
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
        out = ensure_callable(_sigmoid)(ensure_callable(Tensor)(ensure_callable(DummyTensor2D)(), ensure_callable(TensorConfig)(shape=(2, 2), dtype=ensure_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)))
        assert ensure_eager_tensor(out).shape == (2, 2)
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = orig_eager
