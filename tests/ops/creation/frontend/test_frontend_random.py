from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_random import manual_seed, rand, randint, randn


def test_frontend_random():
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            return op

    orig_reg_backend = getattr(reg, "_ACTIVE_BACKEND", None)
    has_reg_backend = hasattr(reg, "_ACTIVE_BACKEND")
    reg._ACTIVE_BACKEND = DummyBackend()
    import ml_switcheroo_compiler.ops.creation.frontend_random as rand_mod

    orig_rand_backend = rand_mod.get_active_backend
    rand_mod.get_active_backend = lambda: DummyBackend()

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert rand(1).data == "Rand"
    assert randn(1).data == "Randn"
    assert randint(0, 10, (1,)).data == "Randint"

    manual_seed(42)

    config.eager_mode = False

    with patch.object(rand_mod, "_emit_creation_node", return_value="emitted"):
        assert rand(1) == "emitted"
        assert randn(1) == "emitted"
        assert randint(0, 10, (1,)) == "emitted"

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph") as graph:
            graph.nodes = {}
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node"):
                manual_seed(42)

    config.eager_mode = True
    if has_reg_backend:
        reg._ACTIVE_BACKEND = orig_reg_backend
    else:
        del reg._ACTIVE_BACKEND
    rand_mod.get_active_backend = orig_rand_backend
