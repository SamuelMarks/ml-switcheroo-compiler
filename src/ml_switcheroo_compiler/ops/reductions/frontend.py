"""Reduction operations frontend."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

from .frontend_pool import (
    adaptive_avg_pool2d,
    adaptive_max_pool2d,
    fold,
    fractional_max_pool2d,
    unfold,
)
from .frontend_segment import (
    segment_max,
    segment_mean,
    segment_min,
    segment_prod,
    segment_sum,
    unsorted_segment_max,
    unsorted_segment_mean,
    unsorted_segment_min,
    unsorted_segment_prod,
    unsorted_segment_sqrt_n,
    unsorted_segment_sum,
)
from .frontend_stats import (
    approx_max_k,
    approx_min_k,
    corrcoef,
    correlate,
    cov,
    ctc_loss,
    pmean,
    psum,
)
from .frontend_utils import reduce_window


def sum(a: object, axis: object = None, keepdims: bool = False) -> Tensor:
    """Sum."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Sum", getattr(a, "data", a), axis=axis, keepdims=keepdims)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    return _emit_shape_node("Sum", [a], {"axis": axis, "keepdims": keepdims}, (None,), getattr(a, "dtype", "float32"))


def max(a: object, axis: object = None, keepdims: bool = False) -> Tensor:
    """Max."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Max", getattr(a, "data", a), axis=axis, keepdims=keepdims)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    return _emit_shape_node("Max", [a], {"axis": axis, "keepdims": keepdims}, (None,), getattr(a, "dtype", "float32"))


def min(a: object, axis: object = None, keepdims: bool = False) -> Tensor:
    """Min."""
    if config.eager_mode:
        data = get_active_backend().execute_op("Min", getattr(a, "data", a), axis=axis, keepdims=keepdims)
        return Tensor(data, TensorConfig(getattr(data, "shape", ()), getattr(a, "dtype", "float32"), getattr(a, "device", None)))
    return _emit_shape_node("Min", [a], {"axis": axis, "keepdims": keepdims}, (None,), getattr(a, "dtype", "float32"))


__all__ = [
    "adaptive_avg_pool2d",
    "sum",
    "max",
    "min",
    "adaptive_max_pool2d",
    "approx_max_k",
    "approx_min_k",
    "corrcoef",
    "correlate",
    "cov",
    "ctc_loss",
    "fold",
    "fractional_max_pool2d",
    "pmean",
    "psum",
    "reduce_window",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "unfold",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
]
