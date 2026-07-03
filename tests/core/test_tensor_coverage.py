"""Provides required module functionality."""

from unittest.mock import MagicMock

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_tensor_eval_coverage() -> None:
    """Execute the requested function."""
    config.eager_mode = False

    mock_data = MagicMock(id="n1")
    t = Tensor(mock_data, TensorConfig((1,), DType.Float32, Device("cpu")))

    global_tracing_state.start_tracing()
    global_tracing_state.active_graph.outputs.append("n1")
    t.eval()

    global_tracing_state.active_graph = None
    t.eval()
    global_tracing_state.stop_tracing()
    global_tracing_state.stop_tracing()

    global_tracing_state.stop_tracing()
    config.eager_mode = True
