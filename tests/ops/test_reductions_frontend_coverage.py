"""Provides required module functionality."""

from ml_switcheroo_compiler.core.dtype import DType

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
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


def test_remaining_segment_ops() -> None:
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.ops.reductions import (
        segment_max,
        segment_mean,
        segment_min,
        segment_prod,
        unsorted_segment_max,
        unsorted_segment_mean,
        unsorted_segment_min,
        unsorted_segment_prod,
        unsorted_segment_sqrt_n,
        unsorted_segment_sum,
        approx_max_k,
        approx_min_k,
    )
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    from ml_switcheroo_compiler.tracing import ProxyTensor

    data = Tensor(
        ProxyTensor(id="data_id", shape=(3,), dtype=DType("float32")),
        TensorConfig((3,), DType("float32"), None),
    )
    segment_ids = Tensor(
        ProxyTensor(id="seg_id", shape=(3,), dtype=DType("int32")),
        TensorConfig((3,), DType("int32"), None),
    )

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        segment_max(data, segment_ids)
        segment_mean(data, segment_ids)
        segment_min(data, segment_ids)
        segment_prod(data, segment_ids)

        unsorted_segment_max(data, segment_ids, 2)
        unsorted_segment_mean(data, segment_ids, 2)
        unsorted_segment_min(data, segment_ids, 2)
        unsorted_segment_prod(data, segment_ids, 2)
        unsorted_segment_sqrt_n(data, segment_ids, 2)
        unsorted_segment_sum(data, segment_ids, 2)

        approx_max_k(data, 2)
        approx_min_k(data, 2)
        _tracer.stop_tracing()
