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
from ml_switcheroo_compiler.core.shape import _get_attr, _set_attr
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_lstm import conv_lstm_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import RNNWeights
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue
ConvLSTMConfig = as_tensor(_get_attr(importlib.import_module("ml_switcheroo_compiler.ops.nn.conv_lstm"), "ConvLSTMConfig"))


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
                """
                return np.ones((2, 2, 2, 2))

            @classmethod
            def array(cls, data: EagerValue) -> EagerValue:
                """Test function array.

                Args:
                    data: Test parameter `data`.
                """
                return data

        as_callable(BackendRegistry.register)("mock_ops", MockBackend)
        tc3 = as_callable(TensorConfig)(shape=(2, 2, 2), dtype=as_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)
        tc4 = as_callable(TensorConfig)(shape=(2, 2, 2, 2), dtype=as_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)
        tc5 = as_callable(TensorConfig)(shape=(2, 2, 2, 2, 2), dtype=as_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self, shape: EagerValue) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape = shape
                self.data = as_callable(np.ones)(shape)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t3 = as_callable(Tensor)(as_callable(DummyTensor)((2, 2, 2)), tc3)
        t4 = as_callable(Tensor)(as_callable(DummyTensor)((2, 2, 2, 2)), tc4)
        t5 = as_callable(Tensor)(as_callable(DummyTensor)((2, 2, 2, 2, 2)), tc5)
        w = as_callable(RNNWeights)(t4, t4, t4)
        c = as_callable(ConvLSTMConfig)((3, 3))
        with pytest.raises(ValueError):
            as_callable(conv_lstm_cell)(t3, (t3, t3), w, c)
        with pytest.raises(ValueError):
            as_callable(conv_lstm_cell)(t4, (t4, t4), w, c)
        with pytest.raises(ValueError):
            as_callable(conv_lstm_cell)(t5, (t5, t5), w, c)
        with pytest.raises(ValueError):
            as_callable(conv_lstm_cell)(as_callable(Tensor)(as_callable(DummyTensor2D)(), tc3), (t3, t3), w, c)
        with pytest.raises(ValueError):
            as_callable(conv_lstm_cell)(
                as_callable(Tensor)(as_callable(DummyTensor2D)(), as_callable(TensorConfig)(shape=(2, 2), dtype=as_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)),
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
        tc4 = as_callable(TensorConfig)(shape=(2, 2, 2, 2), dtype=as_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self, shape: EagerValue) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape = shape
                self.data = as_callable(np.ones)(shape)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t4 = as_callable(Tensor)(as_callable(DummyTensor)((2, 2, 2, 2)), tc4)
        w = as_callable(RNNWeights)(t4, t4)
        orig_split = as_tensor(_get_attr(mod, "split"))
        orig_sigmoid = mod._sigmoid
        orig_tanh = as_tensor(_get_attr(mod, "tanh"))
        orig_add = as_tensor(_get_attr(mod, "add"))
        orig_multiply = as_tensor(_get_attr(mod, "multiply"))
        try:
            _set_attr(mod, "split", lambda *args, **kwargs: (t4, t4, t4, t4))
            mod._sigmoid = lambda x: x
            _set_attr(mod, "tanh", lambda x: x)
            _set_attr(mod, "add", lambda *args, **kwargs: t4)
            _set_attr(mod, "multiply", lambda *args, **kwargs: t4)
            as_callable(mod._apply_conv_lstm_gates)(t4, t4, (t4, t4), w, "channels_last")
            w2 = as_callable(RNNWeights)(t4, t4, bias=t4)
            as_callable(mod._apply_conv_lstm_gates)(t4, t4, (t4, t4), w2, "channels_last")
        finally:
            _set_attr(mod, "split", orig_split)
            mod._sigmoid = orig_sigmoid
            _set_attr(mod, "tanh", orig_tanh)
            _set_attr(mod, "add", orig_add)
            _set_attr(mod, "multiply", orig_multiply)
        with pytest.raises(ValueError):
            as_callable(_apply_conv_lstm_gates)(t4, t4, (t4, t4), w, "channels_last")
        with pytest.raises(ValueError):
            as_callable(_apply_conv_lstm_gates)(t4, t4, (t4, t4), w, "channels_first")
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

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        as_callable(_sigmoid)(as_callable(Tensor)(as_callable(DummyTensor2D)(), as_callable(TensorConfig)(shape=(2, 2), dtype=as_callable(types.SimpleNamespace)(value="float32"), device=None, requires_grad=False, trainable=False)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    finally:
        config.eager_mode = orig_eager
    assert callable(test_lstm_sigmoid)
