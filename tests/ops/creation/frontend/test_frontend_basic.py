# ruff: noqa: E501
from unittest.mock import patch

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_basic import _create_backend_array, _get_dtype_val, _infer_dtype, _try_create_array, array, asarray, convert_to_numpy, convert_to_tensor, empty, empty_like, frombuffer, full, full_like, ones, ones_like, zeros, zeros_like


def test_frontend_basic():

    class DummyItem:
        def item(self):
            return 1

    class DummyData:
        data = DummyItem()

    import numpy as np

    assert _infer_dtype(np.array([1.0])) == DType.Float64
    assert _infer_dtype(np.array([1], dtype="int32")) == DType.Int32
    assert _get_dtype_val(DType.Float32) == "float32"
    assert _get_dtype_val(None) is None

    class DummyBackend:
        def array(self, x, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def asarray(self, x, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

    assert _try_create_array(DummyBackend(), [1], "int32").dtype == "float32"
    assert _try_create_array(DummyBackend(), [1], None).dtype == "float32"

    class DummyBackendThrow:
        def array(self, x, dtype=None):
            raise Exception()

    try:
        _try_create_array(DummyBackendThrow(), [1], None)
    except Exception:
        pass
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend", return_value=DummyBackend()):
        assert _create_backend_array([1], None).dtype == "float32"
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        config.eager_mode = True
        t = array([1])
        assert t.shape == (1,)
        config.eager_mode = False
        import ml_switcheroo_compiler.tracing.state as state

        class DummyGraph:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                pass

        state.global_tracing_state.is_tracing = True
        state.global_tracing_state.active_graph = DummyGraph()
        state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node
        config.eager_mode = False
        array([1])
        asarray([1])
        config.eager_mode = True
        ta = asarray([1])
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        tc = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
        assert convert_to_tensor(tc) is tc
        convert_to_tensor([1])


def test_frontend_basic_functions():
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def array(self, x, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def asarray(self, x, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def zeros(self, s, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def ones(self, s, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def full(self, s, v, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def empty(self, s, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def frombuffer(self, s, dtype=None, count=None, offset=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

    reg._ACTIVE_BACKEND = DummyBackend()
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    zeros((1,))
    ones((1,))
    full((1,), 0)
    empty((1,))
    import numpy as np

    frombuffer(np.zeros(10).tobytes(), count=10)
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    config.eager_mode = False
    config.eager_mode = False
    zeros((1,))
    ones((1,))
    full((1,), 0)
    empty((1,))
    frombuffer(np.zeros(10).tobytes(), count=10)
    assert convert_to_numpy(zeros((1,))) is not None
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
    config.eager_mode = False
    config.eager_mode = False
    zeros_like(t)
    ones_like(t)
    full_like(t, 1)
    empty_like(t)


def test_frontend_basic_missing_branches():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class DummyItem2:
        def item(self):
            return "item"

        def __int__(self):
            return 1

    class DummyTensor2:
        data = DummyItem2()

    import numpy as np

    _infer_dtype(np.array([1], dtype="float16"))
    _infer_dtype(np.array([1], dtype="int16"))
    _infer_dtype(np.array([1], dtype="int8"))
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend") as m:
        m.return_value.array.side_effect = Exception("test")
        try:
            _create_backend_array([1], None)
        except Exception:
            pass
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic.get_active_backend") as m:
        m.return_value.asarray.side_effect = Exception("test")

        class DummyTensor3:
            pass

        try:
            convert_to_tensor([1])
        except Exception:
            pass
    config.eager_mode = True
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend4:
        def empty(self, s, dtype=None):

            class T:
                shape = (1,)
                dtype = "float32"

            return T()

    reg._ACTIVE_BACKEND = DummyBackend4()
    empty((1,))


def test_frontend_basic_convert():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(data=[1], config=TensorConfig((1,), DType.Int32, "cpu"))

    class DummyBackend4:
        def convert_to_numpy(self, x):
            return "numpy_arr"

    import ml_switcheroo_compiler.backends.registry as reg

    reg._ACTIVE_BACKEND = DummyBackend4()
    convert_to_numpy(t)


def test_frontend_basic_kwargs():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        def zeros(self, s, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def ones(self, s, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def full(self, s, v, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def empty(self, s, dtype=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

        def frombuffer(self, s, dtype=None, count=None, offset=None):

            class T:
                dtype = "float32"
                shape = (1,)

            return T()

    reg._ACTIVE_BACKEND = DummyBackend()
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node
    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic._emit_creation_node", return_value="emitted"):
        zeros((1,), dtype=DType.Float32)
        ones((1,), dtype=DType.Float32)
        full((1,), 0, dtype=DType.Float32)
        empty((1,), dtype=DType.Float32)
        frombuffer(b"test", dtype=DType.Float32)
    config.eager_mode = True


def test_frontend_basic_coverage():
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend_basic import _extract_fill_value, _unpack_shape, empty, empty_like, frombuffer, full, full_like, ones, ones_like, zeros, zeros_like

    assert _unpack_shape((1,)) == (1,)

    class DummyItem:
        def item(self):
            return 1

    class DummyData:
        data = DummyItem()

    assert _unpack_shape((DummyData(), DummyItem(), 2)) == (1, 1, 2)
    assert _extract_fill_value(1) == 1
    assert _extract_fill_value(DummyData()) == 1
    assert _extract_fill_value(DummyItem()) == 1
    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            return op

    reg._ACTIVE_BACKEND = DummyBackend()
    pass
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.creation.frontend_basic._emit_creation_node", return_value="emitted"):
        assert zeros((1,), dtype=DType.Float32) == "emitted"
        assert ones((1,), dtype=DType.Float32) == "emitted"
        assert full((1,), 0, dtype=DType.Float32) == "emitted"
        assert empty((1,), dtype=DType.Float32) == "emitted"
        assert frombuffer(np.zeros(10).tobytes(), count=10) == "emitted"
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        t = Tensor(data="data", config=TensorConfig((1,), DType.Float32, "cpu"))
        assert zeros_like(t) == "emitted"
        assert ones_like(t) == "emitted"
        assert full_like(t, 1) == "emitted"
        assert empty_like(t) == "emitted"
    config.eager_mode = True
