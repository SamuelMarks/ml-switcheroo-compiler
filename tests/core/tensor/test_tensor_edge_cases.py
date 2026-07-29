import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import ArrayAtIndexer, Parameter, Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_tensor_coverage():
    config.eager_mode = True
    global_tracing_state.is_tracing = False
    cfg = TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu"))
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), cfg)
    indexer = ArrayAtIndexer(t)
    at_obj = indexer[0]
    assert at_obj.add(1) is t
    assert at_obj.multiply(1) is t
    assert at_obj.set(1) is t
    assert at_obj.maximum(1) is t
    assert at_obj.minimum(1) is t

    assert t.ndim == 2
    assert t.size == 4
    assert t.shape == (2, 2)
    assert t.dtype == DType("float32")
    assert t.device is not None
    assert t.requires_grad is False
    assert (t.data == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    assert t.eval() is t

    class DummyData:
        id = "dummy"

    t_lazy = Tensor(DummyData(), cfg)

    original_eager = config.eager_mode
    original_tracing = global_tracing_state.is_tracing
    original_graph = global_tracing_state.active_graph
    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = True

        class DummyGraph:
            outputs = []
            nodes = {}
            name = "dummy"

        global_tracing_state.active_graph = DummyGraph()
        t_lazy.eval()
        assert "dummy" in global_tracing_state.active_graph.outputs
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = original_tracing
        global_tracing_state.active_graph = original_graph

    import ml_switcheroo_compiler.ops.registry as registry

    def dummy_backward(*args, **kwargs):
        pass

    try:
        old_backward = registry.get_util("backward")
        has_old = True
    except KeyError:
        has_old = False

    registry.register_util("backward")(dummy_backward)
    t.backward()
    if has_old:
        registry.register_util("backward")(old_backward)
    else:
        del registry._UTIL_REGISTRY["backward"]

    def dummy_reshape(*args, **kwargs):
        return args[0]

    registry.register_frontend("reshape")(dummy_reshape)
    assert t.view(4) is t
    assert t.view([2, 2]) is t
    assert t.view((2, 2)) is t

    assert t.contiguous() is t

    t_detached = t.detach()
    assert t_detached.device is not None

    v = Variable(np.array(1.0), TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu"), trainable=True))
    assert v.trainable is True

    t_val = Tensor(np.array(2.0), TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))
    v.assign(t_val)
    v.assign_add(t_val)
    v.assign_sub(t_val)

    original_eager = config.eager_mode
    try:
        config.eager_mode = False

        def dummy_emit_shape_node(*args, **kwargs):
            pass

        registry.register_util("_emit_shape_node")(dummy_emit_shape_node)
        v.assign(t_val)
        v.assign_add(t_val)
        v.assign_sub(t_val)
    finally:
        config.eager_mode = original_eager

    p = Parameter(np.array(2.0), TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))
    assert p.trainable is True
    assert p.__index__() == 2


def test_tensor_mixins_coverage():
    config.eager_mode = True
    global_tracing_state.is_tracing = False
    cfg = TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu"))
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), cfg)

    t_sym = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig(shape=("a", 2), dtype=DType("float32"), device=Device("cpu")))
    pass

    pass
    assert isinstance(np.array(t), np.ndarray)

    class DummyLazyData:
        id = "dummy"

    t_lazy2 = Tensor(DummyLazyData(), cfg)
    assert isinstance(np.array(t_lazy2), np.ndarray)

    class DummyDataList:
        def tolist(self):
            return [1, 2]

    t_list = Tensor(DummyDataList(), TensorConfig(shape=(2,), dtype=DType("float32"), device=Device("cpu")))
    assert isinstance(np.array(t_list), np.ndarray)

    t_scalar = Tensor(np.array(42.0), TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))
    assert t_scalar.item() == 42.0
    assert int(t_scalar) == 42
    assert t_scalar.__index__() == 42
    assert float(t_scalar) == 42.0

    assert bool(Tensor(np.array(True), TensorConfig(shape=(), dtype=DType("bool"), device=Device("cpu")))) is True
    with pytest.raises(ValueError):
        bool(t)

    assert len(t) == 2
    assert len(t_scalar) == 0

    assert len(list(t)) == 2
    with pytest.raises(TypeError):
        list(t_scalar)

    assert t[0].shape == (2,)

    class DummyKeyData:
        data = 0

    assert t[DummyKeyData()].shape == (2,)

    class DummyKeyDataTuple:
        data = (0,)

    assert t[(DummyKeyData(),)].shape == (2,)

    original_eager = config.eager_mode
    original_tracing = global_tracing_state.is_tracing
    original_graph = global_tracing_state.active_graph
    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        t_lazy_idx = t[0]
        assert t_lazy_idx.dtype == DType("float32")

        global_tracing_state.is_tracing = False
        with pytest.raises(RuntimeError):
            t[0]
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = original_tracing
        global_tracing_state.active_graph = original_graph

    t[0] = np.array([5.0, 6.0])

    class DummyValData:
        data = np.array([7.0, 8.0])

    t[0] = DummyValData()

    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        with pytest.raises(TypeError):
            t[0] = 1
    finally:
        config.eager_mode = original_eager

    assert isinstance(t.at, ArrayAtIndexer)
