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
    sum: object = get_op("Sum")()
except KeyError:
    sum: object = None
try:
    prod: object = get_op("Prod")()
except KeyError:
    prod: object = None
try:
    mean: object = get_op("Mean")()
except KeyError:
    mean: object = None
try:
    variance: object = get_op("Variance")()
except KeyError:
    variance: object = None
try:
    std: object = get_op("Std")()
except KeyError:
    std: object = None
try:
    max: object = get_op("Max")()
except KeyError:
    max: object = None
try:
    min: object = get_op("Min")()
except KeyError:
    min: object = None
try:
    argmax: object = get_op("Argmax")()
except KeyError:
    argmax: object = None
try:
    argmin: object = get_op("Argmin")()
except KeyError:
    argmin: object = None
try:
    all: object = get_op("All")()
except KeyError:
    all: object = None
try:
    any: object = get_op("Any")()
except KeyError:
    any: object = None

try:
    logsumexp: object = get_op("Logsumexp")()
except KeyError:
    logsumexp: object = None
try:
    nanargmax: object = get_op("Nanargmax")()
except KeyError:
    nanargmax: object = None
try:
    nanargmin: object = get_op("Nanargmin")()
except KeyError:
    nanargmin: object = None
try:
    nancumprod: object = get_op("Nancumprod")()
except KeyError:
    nancumprod: object = None
try:
    nancumsum: object = get_op("Nancumsum")()
except KeyError:
    nancumsum: object = None
try:
    nanmax: object = get_op("Nanmax")()
except KeyError:
    nanmax: object = None
try:
    nanmean: object = get_op("Nanmean")()
except KeyError:
    nanmean: object = None
try:
    nanmedian: object = get_op("Nanmedian")()
except KeyError:
    nanmedian: object = None
try:
    nanmin: object = get_op("Nanmin")()
except KeyError:
    nanmin: object = None
try:
    nanpercentile: object = get_op("Nanpercentile")()
except KeyError:
    nanpercentile: object = None
try:
    nanprod: object = get_op("Nanprod")()
except KeyError:
    nanprod: object = None
try:
    nanquantile: object = get_op("Nanquantile")()
except KeyError:
    nanquantile: object = None
try:
    nanstd: object = get_op("Nanstd")()
except KeyError:
    nanstd: object = None
try:
    nansum: object = get_op("Nansum")()
except KeyError:
    nansum: object = None
try:
    nanvar: object = get_op("Nanvar")()
except KeyError:
    nanvar: object = None


try:
    count_nonzero: object = get_op("CountNonzero")()
except KeyError:
    count_nonzero: object = None
try:
    norm: object = get_op("Norm")()
except KeyError:
    norm: object = None
try:
    cumsum: object = get_op("Cumsum")()
except KeyError:
    cumsum: object = None

try:
    cummax: object = get_op("Cummax")()
except KeyError:
    cummax: object = None
try:
    cummin: object = get_op("Cummin")()
except KeyError:
    cummin: object = None

try:
    cumprod: object = get_op("Cumprod")()
except KeyError:
    cumprod: object = None


try:
    logcumsumexp: object = get_op("Logcumsumexp")()
except KeyError:
    logcumsumexp: object = None


try:
    accumulate_n: object = get_op("AccumulateN")()
except KeyError:
    accumulate_n: object = None
try:
    add_n: object = get_op("AddN")()
except KeyError:
    add_n: object = None
try:
    cumulative_logsumexp: object = get_op("CumulativeLogsumexp")()
except KeyError:
    cumulative_logsumexp: object = None
try:
    reduce_euclidean_norm: object = get_op("ReduceEuclideanNorm")()
except KeyError:
    reduce_euclidean_norm: object = None
try:
    reduce_logsumexp: object = get_op("Logsumexp")()
except KeyError:
    reduce_logsumexp: object = None
