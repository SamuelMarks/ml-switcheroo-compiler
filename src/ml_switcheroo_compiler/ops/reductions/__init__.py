"""Reductions operations package."""

from __future__ import annotations

from types import ModuleType as _ModuleType

import ml_switcheroo_compiler.ops.reductions.aggregations as _aggregations
import ml_switcheroo_compiler.ops.reductions.api_pool as _api_pool
import ml_switcheroo_compiler.ops.reductions.api_stats as _api_stats
import ml_switcheroo_compiler.ops.reductions.api_utils as _api_utils
import ml_switcheroo_compiler.ops.reductions.boolean as _boolean
import ml_switcheroo_compiler.ops.reductions.core as _core
import ml_switcheroo_compiler.ops.reductions.distributed as _distributed
import ml_switcheroo_compiler.ops.reductions.nan as _nan
import ml_switcheroo_compiler.ops.reductions.pooling as _pooling
import ml_switcheroo_compiler.ops.stats.descriptive as _statistical
from ml_switcheroo_compiler.ops.base import OpDef, get_op

from .api import adaptive_avg_pool2d as adaptive_avg_pool2d
from .api import adaptive_max_pool2d as adaptive_max_pool2d
from .api import approx_max_k as approx_max_k
from .api import approx_min_k as approx_min_k
from .api import corrcoef as corrcoef
from .api import correlate as correlate
from .api import cov as cov
from .api import pmean as pmean
from .api import psum as psum
from .api import reduce_window as reduce_window
from .api import segment_max as segment_max
from .api import segment_mean as segment_mean
from .api import segment_min as segment_min
from .api import segment_prod as segment_prod
from .api import segment_sum as segment_sum
from .api import unsorted_segment_max as unsorted_segment_max
from .api import unsorted_segment_mean as unsorted_segment_mean
from .api import unsorted_segment_min as unsorted_segment_min
from .api import unsorted_segment_prod as unsorted_segment_prod
from .api import unsorted_segment_sqrt_n as unsorted_segment_sqrt_n
from .api import unsorted_segment_sum as unsorted_segment_sum
from .distributed import Pmean as Pmean
from .distributed import Psum as Psum

_: _ModuleType = _nan
_ = _pooling
_ = _aggregations
_ = _api_pool
_ = _api_stats
_ = _api_utils
_ = _boolean
_ = _core
_ = _distributed
_ = _statistical


def _safe_get_op(name: str) -> OpDef:
    """Instantiate a registered reduction op or return a fallback `OpDef` if missing.

    Args:
        name (str): Registered operation name.

    Returns:
        OpDef: Instantiated operation definition.
    """
    try:
        return get_op(name)()
    except KeyError:
        op = OpDef()
        op.op_type = name
        return op


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
