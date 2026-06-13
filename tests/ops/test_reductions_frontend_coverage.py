"""Provides required module functionality."""

from ml_switcheroo_compiler.ops.reductions.frontend import segment_sum
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import config


def test_reductions_frontend_coverage_brute() -> None:
    """Execute the requested function."""
    config.eager_mode = False

    from ml_switcheroo_compiler.tracing.tracer import _tracer, ProxyTensor

    _tracer.start_tracing()
    data = Tensor(
        data=ProxyTensor(id="data_id", shape=(5,)),
        shape=(5,),
        dtype=DType.Float32,
        device=Device("cpu"),
    )
    segment_ids = Tensor(
        data=ProxyTensor(id="segment_id", shape=(5,)),
        shape=(5,),
        dtype=DType.Int32,
        device=Device("cpu"),
    )

    segment_sum(data, segment_ids)  # Test branch where num_segments is None

    _tracer.stop_tracing()

    config.eager_mode = True
