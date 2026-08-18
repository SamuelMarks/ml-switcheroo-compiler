"""Module test_random_state.py."""


def test_random_state():
    """Module test_random_state.py."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.random.state import _dispatch_random, _dispatch_random_eager, _emit_random_node, _get_numpy_rng, rng_uniform
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    t = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", "cpu"))

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = "mock_val"
    mock_backend.get_numpy_rng.return_value = "rng"

    orig_eager = config.eager_mode
    try:
        with patch("ml_switcheroo_compiler.random.state.get_active_backend", return_value=mock_backend):
            config.eager_mode = True

            res = _emit_random_node("SomeRandom", [Tensor(ProxyTensor(id="id", shape=(2,), dtype="float32"), TensorConfig((2,), "float32", "cpu"))], (2,), "float32")
            assert getattr(res, "data", res) == "mock_val"

            res = _dispatch_random_eager("some_func", "SomeRandom", t)
            assert res == "mock_val"

            res = _dispatch_random("some_func", t)
            assert res == "mock_val"

            res = rng_uniform(0, 1, (2,), "float32")
            assert res == "mock_val"

        with patch("ml_switcheroo_compiler.backends.registry.BackendRegistry.get", return_value=mock_backend):
            assert _get_numpy_rng() == "rng"

        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
            with patch("ml_switcheroo_compiler.random.state.get_op", return_value=None):
                res = _dispatch_random("missing_op", t)
                assert res == "emitted"

            mock_op_cls = MagicMock()
            mock_op_inst = MagicMock(return_value="called_op")
            mock_op_cls.return_value = mock_op_inst
            with patch("ml_switcheroo_compiler.random.state.get_op", return_value=mock_op_cls):
                assert _dispatch_random("some_op", t) == "called_op"

        # test _emit_random_node in eager=False
        from ml_switcheroo_compiler.tracing.tracer import global_tracing_state

        global_tracing_state.start_tracing()
        try:
            res = _emit_random_node("SomeRandom", [Tensor(ProxyTensor(id="id", shape=(2,), dtype="float32"), TensorConfig((2,), "float32", "cpu"))], (2,), "float32")
            assert res.shape == (2,)
        finally:
            global_tracing_state.stop_tracing()

    finally:
        config.eager_mode = orig_eager


def test_random_state_eager_with_attrs():
    """test_random_state_eager_with_attrs."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.random.state import _emit_random_node

    t = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", "cpu"))

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = "mock_val"

    orig_eager = config.eager_mode
    try:
        with patch("ml_switcheroo_compiler.random.state.get_active_backend", return_value=mock_backend):
            config.eager_mode = True

            res = _emit_random_node("SomeRandom", [t], (2,), "float32", attributes={"shape": (2,), "dtype": "float32"})
            assert getattr(res, "data", res) == "mock_val"
    finally:
        config.eager_mode = orig_eager
