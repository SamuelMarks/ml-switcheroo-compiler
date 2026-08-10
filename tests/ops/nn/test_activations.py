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
