from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.creation.frontend_matrix import _diag_eager, diag, eye, identity


def test_frontend_matrix():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            class T:
                shape = (1, 1)

            return T()

    import ml_switcheroo_compiler.backends.registry as reg

    reg._ACTIVE_BACKEND = DummyBackend()
    import ml_switcheroo_compiler.ops.creation.frontend_matrix

    ml_switcheroo_compiler.ops.creation.frontend_matrix.get_active_backend = lambda: DummyBackend()

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert eye(1).shape == (1, 1)
    assert identity(1).shape == (1, 1)

    class DummyInput:
        data = "data"
        device = "cpu"
        dtype = "float32"
        shape = (1,)

    class DummyInputNoDtype:
        data = "data"
        device = "cpu"
        dtype = None
        shape = (1,)

    assert _diag_eager(DummyInput(), 0, "cpu", "float32").shape == (1, 1)
    assert _diag_eager(DummyInputNoDtype(), 0, "cpu", None).shape == (1, 1)

    assert diag(DummyInput()).shape == (1, 1)

    class DummyInput2:
        data = type("D", (), {"id": "id"})()
        device = "cpu"
        dtype = "float32"
        shape = (2, 2)

    config.eager_mode = False
    with pytest.raises(RuntimeError):
        diag(DummyInput2())

    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node

    assert diag(DummyInput2()).shape == (2,)

    with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix._emit_creation_node", return_value="emitted"):
        assert eye(1) == "emitted"

    config.eager_mode = True

    class DummyInput3:
        data = type("D", (), {"id": "id"})()
        device = "cpu"
        dtype = "float32"
        shape = (1, 1, 1)

    config.eager_mode = False
    with pytest.raises(ValueError):
        diag(DummyInput3())
