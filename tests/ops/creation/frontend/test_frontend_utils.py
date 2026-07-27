import pytest

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_creation_node


def test_frontend_utils():
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

    config.eager_mode = False
    t = _emit_creation_node("zeros", (1,), DType.Float32, {"a": 1})
    pass

    state.global_tracing_state.is_tracing = False
    with pytest.raises(RuntimeError):
        _emit_creation_node("zeros", (1,), DType.Float32, {"a": 1})
    state.global_tracing_state.is_tracing = True


def test_frontend_utils_constant():
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_constant_node

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        def array(self, value, dtype=None):
            class T:
                shape = (1,)
                ndim = 1

                def tolist(self):
                    return [value]

            return T()

    reg._ACTIVE_BACKEND = DummyBackend()

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    t = _emit_constant_node(0, DType.Float32)
    pass

    state.global_tracing_state.is_tracing = False
    with pytest.raises(RuntimeError):
        _emit_constant_node(0, DType.Float32)
    state.global_tracing_state.is_tracing = True


def test_frontend_utils_constant_scalar():
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_constant_node

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        def array(self, value, dtype=None):
            class T:
                shape = ()
                ndim = 0

                def item(self):
                    return value

            return T()

    reg._ACTIVE_BACKEND = DummyBackend()

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    _emit_constant_node(0, DType.Float32)
