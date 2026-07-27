from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_random import manual_seed, rand, randint, randn


def test_frontend_random():
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            return op

    reg._ACTIVE_BACKEND = DummyBackend()
    import ml_switcheroo_compiler.ops.creation.frontend_random as rand_mod

    rand_mod.get_active_backend = lambda: DummyBackend()

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert rand(1).data == "Rand"
    assert randn(1).data == "Randn"
    assert randint(0, 10, (1,)).data == "Randint"

    manual_seed(42)

    config.eager_mode = False

    with patch("ml_switcheroo_compiler.ops.creation.frontend_random._emit_creation_node", return_value="emitted"):
        assert rand(1) == "emitted"
        assert randn(1) == "emitted"
        assert randint(0, 10, (1,)) == "emitted"

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph") as graph:
            graph.nodes = {}
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node"):
                manual_seed(42)

    config.eager_mode = True
