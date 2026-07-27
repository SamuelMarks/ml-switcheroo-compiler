from unittest.mock import patch

from ml_switcheroo_compiler.ops.creation.frontend_sequence import arange, linspace


def test_frontend_sequence():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            class T:
                shape = (1,)

            return T()

    import ml_switcheroo_compiler.ops.creation.frontend_sequence

    ml_switcheroo_compiler.ops.creation.frontend_sequence.get_active_backend = lambda: DummyBackend()

    assert arange(10).shape == (10,)
    assert arange(0, 10).shape == (10,)
    assert linspace(1, 10, 50).shape == (50,)

    config.eager_mode = False

    with patch("ml_switcheroo_compiler.ops.creation.frontend_sequence._emit_creation_node", return_value="emitted"):
        assert arange(10) == "emitted"
        assert arange(0, 10) == "emitted"
        assert linspace(1, 10, 50) == "emitted"

    config.eager_mode = True
