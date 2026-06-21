"""Provides required module functionality."""

from unittest.mock import MagicMock

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_tensor_eval_coverage() -> None:
    """Execute the requested function."""
    config.eager_mode = False

    from ml_switcheroo_compiler.tracing.tracer import _tracer

    mock_data = MagicMock(id="n1")
    t = Tensor(mock_data, TensorConfig((1,), DType.Float32, Device("cpu")))

    _tracer.start_tracing()
    _tracer.active_graph.outputs.append("n1")
    t.eval()

    _tracer.active_graph = None
    t.eval()

    _tracer.stop_tracing()
    config.eager_mode = True
