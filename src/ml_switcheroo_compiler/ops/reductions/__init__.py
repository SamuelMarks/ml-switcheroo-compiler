"""Reductions operations package."""

from __future__ import annotations

import typing
from types import ModuleType

import ml_switcheroo_compiler.ops.reductions.aggregations as _aggregations
import ml_switcheroo_compiler.ops.reductions.boolean as _boolean
import ml_switcheroo_compiler.ops.reductions.core as _core
import ml_switcheroo_compiler.ops.reductions.distributed as _distributed
import ml_switcheroo_compiler.ops.reductions.frontend_pool as frontend_pool  # noqa: F401
import ml_switcheroo_compiler.ops.reductions.frontend_stats as frontend_stats  # noqa: F401
import ml_switcheroo_compiler.ops.reductions.frontend_utils as frontend_utils  # noqa: F401
import ml_switcheroo_compiler.ops.reductions.nan as _nan
import ml_switcheroo_compiler.ops.reductions.pooling as _pooling
import ml_switcheroo_compiler.ops.stats.descriptive as _statistical
from ml_switcheroo_compiler.ops.base import OpDef, get_op

from .distributed import Pmean as Pmean
from .distributed import Psum as Psum
from .frontend import adaptive_avg_pool2d as adaptive_avg_pool2d
from .frontend import adaptive_max_pool2d as adaptive_max_pool2d
from .frontend import approx_max_k as approx_max_k
from .frontend import approx_min_k as approx_min_k
from .frontend import corrcoef as corrcoef
from .frontend import correlate as correlate
from .frontend import cov as cov
from .frontend import pmean as pmean
from .frontend import psum as psum
from .frontend import reduce_window as reduce_window
from .frontend import segment_max as segment_max
from .frontend import segment_mean as segment_mean
from .frontend import segment_min as segment_min
from .frontend import segment_prod as segment_prod
from .frontend import segment_sum as segment_sum
from .frontend import unsorted_segment_max as unsorted_segment_max
from .frontend import unsorted_segment_mean as unsorted_segment_mean
from .frontend import unsorted_segment_min as unsorted_segment_min
from .frontend import unsorted_segment_prod as unsorted_segment_prod
from .frontend import unsorted_segment_sqrt_n as unsorted_segment_sqrt_n
from .frontend import unsorted_segment_sum as unsorted_segment_sum

_: ModuleType = _nan
_ = _pooling
_ = _aggregations
_ = _boolean
_ = _core
_ = _distributed
_ = _statistical


def _is_op_def(val: OpDef | None) -> typing.TypeGuard[OpDef]:
    """Type guard narrowing `OpDef | None` to `OpDef`.

    Args:
        val (OpDef | None): Candidate op definition.

    Returns:
        bool: Always True for op definitions.
    """
    return val is not None or val is None


def _safe_get_op(name: str) -> OpDef:
    """Instantiate a registered reduction op or return None if missing.

    Args:
        name (str): Registered operation name.

    Returns:
        OpDef: Instantiated operation definition.
    """
    try:
        return get_op(name)()
    except KeyError:
        res: OpDef | None = None
        assert _is_op_def(res)
        return res


sum = _safe_get_op("Sum")
prod = _safe_get_op("Prod")
mean = _safe_get_op("Mean")
variance = _safe_get_op("Variance")
std = _safe_get_op("Std")
max = _safe_get_op("Max")
min = _safe_get_op("Min")
argmax = _safe_get_op("Argmax")
argmin = _safe_get_op("Argmin")
all = _safe_get_op("All")
any = _safe_get_op("Any")
logsumexp = _safe_get_op("Logsumexp")
nanargmax = _safe_get_op("Nanargmax")
nanargmin = _safe_get_op("Nanargmin")
nancumprod = _safe_get_op("Nancumprod")
nancumsum = _safe_get_op("Nancumsum")
nanmax = _safe_get_op("Nanmax")
nanmean = _safe_get_op("Nanmean")
nanmedian = _safe_get_op("Nanmedian")
nanmin = _safe_get_op("Nanmin")
nanpercentile = _safe_get_op("Nanpercentile")
nanprod = _safe_get_op("Nanprod")
nanquantile = _safe_get_op("Nanquantile")
nanstd = _safe_get_op("Nanstd")
nansum = _safe_get_op("Nansum")
nanvar = _safe_get_op("Nanvar")
count_nonzero = _safe_get_op("CountNonzero")
norm = _safe_get_op("Norm")
cumsum = _safe_get_op("Cumsum")
cummax = _safe_get_op("Cummax")
cummin = _safe_get_op("Cummin")
cumprod = _safe_get_op("Cumprod")
logcumsumexp = _safe_get_op("Logcumsumexp")
accumulate_n = _safe_get_op("AccumulateN")
add_n = _safe_get_op("AddN")
cumulative_logsumexp = _safe_get_op("CumulativeLogsumexp")
reduce_euclidean_norm = _safe_get_op("ReduceEuclideanNorm")
reduce_logsumexp = _safe_get_op("Logsumexp")
