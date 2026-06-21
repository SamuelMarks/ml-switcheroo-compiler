"""Provides required module functionality."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend import segment_sum


def test_reductions_frontend_coverage_brute() -> None:
    """Execute the requested function."""
    config.eager_mode = False

    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    _tracer.start_tracing()
    data = Tensor(
        ProxyTensor(id="data_id", shape=(5,)), TensorConfig((5,), DType.Float32, Device("cpu"))
    )
    segment_ids = Tensor(
        ProxyTensor(id="segment_id", shape=(5,)), TensorConfig((5,), DType.Int32, Device("cpu"))
    )

    segment_sum(data, segment_ids)  # Test branch where num_segments is None

    _tracer.stop_tracing()

    config.eager_mode = True
