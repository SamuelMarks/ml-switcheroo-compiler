import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.activations import LogSoftmax, OneHot, Rrelu, Sigmoid, Softmax, crelu, log_softmax, one_hot, rrelu, sigmoid, softmax, softplus


def test_nn_activations_coverage():
    config.eager_mode = True
    t = Tensor(np.array([-1.0, 0.0, 1.0]), TensorConfig(shape=(3,), dtype=DType("float32"), device=Device("cpu")))

    assert crelu(t) is not None

    assert softplus(t) is not None

    # functions eager
    assert softmax(t) is not None
    assert log_softmax(t) is not None
    assert sigmoid(t) is not None
    t_idx = Tensor(np.array([0, 1]), TensorConfig(shape=(2,), dtype=DType("int32"), device=Device("cpu")))
    assert one_hot(t_idx, 2) is not None
    assert rrelu(t) is not None

    # OpDefs
    class DummyShape:
        shape = (1, 2)

    assert Softmax().infer_shape(DummyShape()) == (1, 2)
    assert Softmax().infer_shape() == ()
    assert LogSoftmax().infer_shape(DummyShape()) == (1, 2)
    assert LogSoftmax().infer_shape() == ()
    assert Sigmoid().infer_shape(DummyShape()) == (1, 2)
    assert Sigmoid().infer_shape() == ()
    assert OneHot().infer_shape(DummyShape(), 5) == (1, 2, 5)
    assert OneHot().infer_shape(DummyShape(), depth=5) == (1, 2, 5)
    assert OneHot().infer_shape() == (1,)
    assert Rrelu().infer_shape(DummyShape()) == (1, 2)
    assert Rrelu().infer_shape() == ()

    from ml_switcheroo_compiler.ops.nn.activations import HardSilu, HardSwish, Squareplus

    assert HardSilu().infer_shape(DummyShape(), DummyShape()) == (1, 2)
    assert HardSilu().infer_shape() == ()
    assert HardSwish().infer_shape(DummyShape(), DummyShape()) == (1, 2)
    assert HardSwish().infer_shape() == ()
    assert Squareplus().infer_shape(DummyShape(), DummyShape()) == (1, 2)
    assert Squareplus().infer_shape() == ()
    assert Softmax().infer_shape(DummyShape(), DummyShape()) == (1, 2)
    assert LogSoftmax().infer_shape(DummyShape(), DummyShape()) == (1, 2)
    assert Sigmoid().infer_shape(DummyShape(), DummyShape()) == (1, 2)
    assert Rrelu().infer_shape(DummyShape(), DummyShape()) == (1, 2)

    # tracing
    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        res = softmax(t)
        assert res is not None
        res = log_softmax(t)
        assert res is not None
        res = sigmoid(t)
        assert res is not None
        res = one_hot(t_idx, 2)
        assert res is not None
        res = rrelu(t)
        assert res is not None

    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = False
