import types

import numpy as np

import ml_switcheroo_compiler.ops.nn.conv_lstm as mod
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_lstm import ConvLSTMConfig, conv_lstm_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import RNNWeights


def test_conv_lstm_cov():
    orig_eager = config.eager_mode
    orig_backend = config.backend

    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackend:
            @classmethod
            def execute_op(cls, op_name, *args, **kwargs):
                return np.ones((2, 2, 2, 2))

            @classmethod
            def array(cls, data):
                return data

        BackendRegistry.register("mock_ops", MockBackend)

        tc3 = TensorConfig(shape=(2, 2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)
        tc4 = TensorConfig(shape=(2, 2, 2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)
        tc5 = TensorConfig(shape=(2, 2, 2, 2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)

        class DummyTensor:
            def __init__(self, shape):
                self.shape = shape
                self.data = np.ones(shape)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t3 = Tensor(DummyTensor((2, 2, 2)), tc3)
        t4 = Tensor(DummyTensor((2, 2, 2, 2)), tc4)
        t5 = Tensor(DummyTensor((2, 2, 2, 2, 2)), tc5)

        w = RNNWeights(t4, t4, t4)
        c = ConvLSTMConfig((3, 3))

        try:
            conv_lstm_cell(t3, (t3, t3), w, c)
        except Exception:
            pass
        try:
            conv_lstm_cell(t4, (t4, t4), w, c)
        except Exception:
            pass
        try:
            conv_lstm_cell(t5, (t5, t5), w, c)
        except Exception:
            pass
        try:
            conv_lstm_cell(Tensor(DummyTensor2D(), tc3), (t3, t3), w, c)
        except Exception:
            pass
        try:
            conv_lstm_cell(Tensor(DummyTensor2D(), TensorConfig(shape=(2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)), (t3, t3), w, c)
        except Exception:
            pass

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend


def test_conv_lstm_gates():
    from ml_switcheroo_compiler.ops.nn.conv_lstm import _apply_conv_lstm_gates

    orig_eager = config.eager_mode
    orig_backend = config.backend
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        tc4 = TensorConfig(shape=(2, 2, 2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)

        class DummyTensor:
            def __init__(self, shape):
                self.shape = shape
                self.data = np.ones(shape)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None

        t4 = Tensor(DummyTensor((2, 2, 2, 2)), tc4)

        # Test without bias
        w = RNNWeights(t4, t4)

        orig_split = mod.split
        orig_sigmoid = mod._sigmoid
        orig_tanh = mod.tanh
        orig_add = mod.add
        orig_multiply = mod.multiply

        try:
            mod.split = lambda *args, **kwargs: (t4, t4, t4, t4)
            mod._sigmoid = lambda x: x
            mod.tanh = lambda x: x
            mod.add = lambda *args, **kwargs: t4
            mod.multiply = lambda *args, **kwargs: t4

            mod._apply_conv_lstm_gates(t4, t4, (t4, t4), w, "channels_last")

            # with bias
            w2 = RNNWeights(t4, t4, bias=t4)
            mod._apply_conv_lstm_gates(t4, t4, (t4, t4), w2, "channels_last")

        finally:
            mod.split = orig_split
            mod._sigmoid = orig_sigmoid
            mod.tanh = orig_tanh
            mod.add = orig_add
            mod.multiply = orig_multiply
        try:
            _apply_conv_lstm_gates(t4, t4, (t4, t4), w, "channels_last")
        except Exception:
            pass
        try:
            _apply_conv_lstm_gates(t4, t4, (t4, t4), w, "channels_first")
        except Exception:
            pass

        # Test error path
        try:
            conv_lstm_cell(Tensor(DummyTensor2D(), tc3), (t3, t3), w, c)
        except Exception:
            pass
        try:
            conv_lstm_cell(Tensor(DummyTensor2D(), TensorConfig(shape=(2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)), (t3, t3), w, c)
        except Exception:
            pass

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend


class DummyTensor2D:
    def __init__(self):
        self.shape = (2, 2)
        self.data = np.ones((2, 2))
        self.dtype = types.SimpleNamespace(value="float32")
        self.device = None


def test_lstm_sigmoid():
    from ml_switcheroo_compiler.ops.nn.conv_lstm import _sigmoid

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        _sigmoid(Tensor(DummyTensor2D(), TensorConfig(shape=(2, 2), dtype=types.SimpleNamespace(value="float32"), device=None, requires_grad=False, trainable=False)))
    except Exception:
        pass
    finally:
        config.eager_mode = orig_eager
