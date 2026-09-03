# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

"""Reductions operations package."""

import ml_switcheroo_compiler.ops.reductions.aggregations as _aggregations
import ml_switcheroo_compiler.ops.reductions.boolean as _boolean
import ml_switcheroo_compiler.ops.reductions.core as _core
import ml_switcheroo_compiler.ops.reductions.distributed as _distributed
import ml_switcheroo_compiler.ops.reductions.frontend_pool as frontend_pool
import ml_switcheroo_compiler.ops.reductions.frontend_stats as frontend_stats
import ml_switcheroo_compiler.ops.reductions.frontend_utils as frontend_utils
import ml_switcheroo_compiler.ops.reductions.nan as _nan
import ml_switcheroo_compiler.ops.reductions.pooling as _pooling
import ml_switcheroo_compiler.ops.stats.descriptive as _statistical
from ml_switcheroo_compiler.ops.base import get_op

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

_ = _nan
_ = _pooling
_ = _aggregations
_ = _boolean
_ = _core
_ = _distributed
_ = _statistical


try:
    sum = get_op("Sum")()
except KeyError:
    sum = None
try:
    prod = get_op("Prod")()
except KeyError:
    prod = None
try:
    mean = get_op("Mean")()
except KeyError:
    mean = None
try:
    variance = get_op("Variance")()
except KeyError:
    variance = None
try:
    std = get_op("Std")()
except KeyError:
    std = None
try:
    max = get_op("Max")()
except KeyError:
    max = None
try:
    min = get_op("Min")()
except KeyError:
    min = None
try:
    argmax = get_op("Argmax")()
except KeyError:
    argmax = None
try:
    argmin = get_op("Argmin")()
except KeyError:
    argmin = None
try:
    all = get_op("All")()
except KeyError:
    all = None
try:
    any = get_op("Any")()
except KeyError:
    any = None

try:
    logsumexp = get_op("Logsumexp")()
except KeyError:
    logsumexp = None
try:
    nanargmax = get_op("Nanargmax")()
except KeyError:
    nanargmax = None
try:
    nanargmin = get_op("Nanargmin")()
except KeyError:
    nanargmin = None
try:
    nancumprod = get_op("Nancumprod")()
except KeyError:
    nancumprod = None
try:
    nancumsum = get_op("Nancumsum")()
except KeyError:
    nancumsum = None
try:
    nanmax = get_op("Nanmax")()
except KeyError:
    nanmax = None
try:
    nanmean = get_op("Nanmean")()
except KeyError:
    nanmean = None
try:
    nanmedian = get_op("Nanmedian")()
except KeyError:
    nanmedian = None
try:
    nanmin = get_op("Nanmin")()
except KeyError:
    nanmin = None
try:
    nanpercentile = get_op("Nanpercentile")()
except KeyError:
    nanpercentile = None
try:
    nanprod = get_op("Nanprod")()
except KeyError:
    nanprod = None
try:
    nanquantile = get_op("Nanquantile")()
except KeyError:
    nanquantile = None
try:
    nanstd = get_op("Nanstd")()
except KeyError:
    nanstd = None
try:
    nansum = get_op("Nansum")()
except KeyError:
    nansum = None
try:
    nanvar = get_op("Nanvar")()
except KeyError:
    nanvar = None


try:
    count_nonzero = get_op("CountNonzero")()
except KeyError:
    count_nonzero = None
try:
    norm = get_op("Norm")()
except KeyError:
    norm = None
try:
    cumsum = get_op("Cumsum")()
except KeyError:
    cumsum = None

try:
    cummax = get_op("Cummax")()
except KeyError:
    cummax = None
try:
    cummin = get_op("Cummin")()
except KeyError:
    cummin = None

try:
    cumprod = get_op("Cumprod")()
except KeyError:
    cumprod = None


try:
    logcumsumexp = get_op("Logcumsumexp")()
except KeyError:
    logcumsumexp = None


try:
    accumulate_n = get_op("AccumulateN")()
except KeyError:
    accumulate_n = None
try:
    add_n = get_op("AddN")()
except KeyError:
    add_n = None
try:
    cumulative_logsumexp = get_op("CumulativeLogsumexp")()
except KeyError:
    cumulative_logsumexp = None
try:
    reduce_euclidean_norm = get_op("ReduceEuclideanNorm")()
except KeyError:
    reduce_euclidean_norm = None
try:
    reduce_logsumexp = get_op("Logsumexp")()
except KeyError:
    reduce_logsumexp = None
