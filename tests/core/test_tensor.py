def test_tensor_array_at():
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import ArrayAt, ArrayAtIndexer, Tensor, TensorConfig

    t = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    at = ArrayAt(t, 0)

    assert at.add(1) is t
    assert at.multiply(2) is t
    assert at.minimum(1) is t
    assert at.maximum(1) is t
    assert at.set(1) is t

    indexer = ArrayAtIndexer(t)
    assert indexer[0].add(1) is t


def test_tensor_eval_view_detach_backward():
    pass


def test_variable_assign():
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable

    v = Variable(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    t = Tensor(np.array([2, 3]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))

    class FakeBackend:
        def execute_op(self, name, data1, data2):
            return data1

    # eager
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=FakeBackend()):
        assert v.assign(t) is v
        assert v.assign_add(t) is v
        assert v.assign_sub(t) is v

    # traced
    config.eager_mode = False

    class FakeData:
        id = "dummy"

    v._data = FakeData()
    t._data = FakeData()

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value=None):
        with patch("ml_switcheroo_compiler.ops.registry.get_util", return_value=lambda *a, **k: None):
            assert v.assign(t) is v
            assert v.assign_add(t) is v
            assert v.assign_sub(t) is v

    config.eager_mode = True


def test_parameter():
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Parameter, TensorConfig

    p = Parameter(np.array([42]), TensorConfig(shape=(1,), dtype="int32", device="cpu"))
    assert p.trainable is True

    # Fake numpy method to avoid execution
    class FakeData:
        def __int__(self):
            return 42

    p.numpy = lambda: 42
    assert p.__index__() == 42


def test_tensor_config_list_shape_and_parse_dim():
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # 126
    c = TensorConfig(shape=[2, "dynamic"], dtype="float32", device="cpu")
    assert isinstance(c.shape, tuple)

    # 172, 173 - will happen when passing "dynamic" to Tensor config shape
    t = Tensor(np.array([1, 2]), c)
    assert t.shape == (2, "dynamic")


def test_tensor_eval_already_in_graph():
    from unittest.mock import MagicMock

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    t = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    config.eager_mode = False
    global_tracing_state.start_tracing()
    global_tracing_state.active_graph = MagicMock()
    global_tracing_state.active_graph.outputs = ["n1"]

    class FakeData:
        id = "n1"

    t._data = FakeData()
    assert t.eval() is t
    config.eager_mode = True
    global_tracing_state.stop_tracing()


def test_tensor_eval_not_tracing():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    t = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    config.eager_mode = False

    class FakeData:
        id = "n1"

    t._data = FakeData()

    # 190, 194 where global_tracing_state.is_tracing is False
    global_tracing_state.is_tracing = False
    assert t.eval() is t
    config.eager_mode = True


def test_tensor_view_list_shape():
    # We want to hit view() with a list/tuple inside shape
    # E.g., t.view([2, 2]) or t.view((2, 2)) vs t.view(2, 2)
    # The shape arg is *shape, so shape = ([2, 2],)
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig(shape=(2, 2), dtype="float32", device="cpu"))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.registry.get_frontend") as mock_get_frontend:
        mock_get_frontend.return_value = lambda x, shape: Tensor(np.zeros(shape), TensorConfig(shape=shape, dtype="float32", device="cpu"))
        res = t.view([4])
        assert res.shape == (4,)

        res2 = t.view(2, 2)
        assert res2.shape == (2, 2)

        res3 = t.view((4,))
        assert res3.shape == (4,)


def test_tensor_eval_branches():
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # eager mode -> returns self
    t1 = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    assert t1.eval() is t1

    # tracing mode
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.tracer import global_tracing_state

    config.eager_mode = False

    class DummyGraph:
        outputs = []

    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = DummyGraph()

    # Create an object with an id
    class DummyData:
        id = "test_id"

    t2 = Tensor(DummyData(), TensorConfig((), "float32", "cpu"))

    # Not in graph outputs -> append
    assert t2.eval() is t2
    assert DummyGraph.outputs == ["test_id"]

    # Already in graph outputs -> no append
    assert t2.eval() is t2
    assert DummyGraph.outputs == ["test_id"]

    # Reset
    config.eager_mode = True
    global_tracing_state.is_tracing = False
    global_tracing_state.active_graph = None


def test_tensor_contiguous():
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    assert t.contiguous() is t


def test_tensor_detach():
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    res = t.detach()
    assert res is not t
    assert isinstance(res, Tensor)


def test_tensor_backward():
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.array(1), TensorConfig((), "float32", "cpu"))
    with patch("ml_switcheroo_compiler.ops.registry.get_util") as mock_get_util:
        mock_func = mock_get_util.return_value
        t.backward(1, test=True)
        mock_func.assert_called_once_with(t, 1, test=True)
        mock_get_util.assert_called_once_with("backward")
