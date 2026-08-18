"""Module test_activations.py."""

from ml_switcheroo_compiler.core.dtype import DType

"""Test nn activations coverage."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.activations import isotonic_regression
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_isotonic_regression_tracing():
    """Test isotonic_regression in tracing mode."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        y = Tensor(None, TensorConfig((5,), "float32", "cpu"))
        out1, out2 = isotonic_regression(y)
        assert out1.shape == (5,)
        assert out2.shape == (5,)
        assert out2.dtype == DType.Int32
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = True


def test_activations_dispatcher():
    """test_activations_dispatcher."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.nn.activations import hard_silu, hard_swish, mish, squareplus

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"

        assert hard_silu(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("HardSilu", 1, kw=2)

        assert hard_swish(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("HardSwish", 1, kw=2)

        assert mish(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("Mish", 1, kw=2)

        assert squareplus(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("Squareplus", 1, kw=2)
