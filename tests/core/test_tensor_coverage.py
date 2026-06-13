"""Provides required module functionality."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.config import config
from unittest.mock import MagicMock


def test_tensor_eval_coverage() -> None:
    """Execute the requested function."""
    config.eager_mode = False

    from ml_switcheroo_compiler.tracing.tracer import _tracer

    mock_data = MagicMock(id="n1")
    t = Tensor(data=mock_data, shape=(1,), dtype=DType.Float32, device=Device("cpu"))

    _tracer.start_tracing()
    _tracer.active_graph.outputs.append("n1")
    t.eval()

    _tracer.active_graph = None
    t.eval()

    _tracer.stop_tracing()
    config.eager_mode = True
