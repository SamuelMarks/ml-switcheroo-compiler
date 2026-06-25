"""Reductions operations package."""

import ml_switcheroo_compiler.ops.reductions.basic as _basic
import ml_switcheroo_compiler.ops.reductions.nan as _nan
import ml_switcheroo_compiler.ops.reductions.pooling as _pooling
import ml_switcheroo_compiler.ops.reductions.aggregations as _aggregations
import ml_switcheroo_compiler.ops.reductions.boolean as _boolean
import ml_switcheroo_compiler.ops.reductions.core as _core
import ml_switcheroo_compiler.ops.reductions.distributed as _distributed
import ml_switcheroo_compiler.ops.reductions.statistical as _statistical
from ml_switcheroo_compiler.ops.reductions.statistical import moments
from ml_switcheroo_compiler.ops.base import get_op

from .distributed import Pmean, Psum
from .aggregations import ReduceWindow
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
from .frontend import approx_max_k as approx_max_k
from .frontend import approx_min_k as approx_min_k

_ = _basic
_ = _nan
_ = _pooling
_ = _aggregations
_ = _boolean
_ = _core
_ = _distributed
_ = _statistical


sum = get_op("Sum")()
prod = get_op("Prod")()
mean = get_op("Mean")()
variance = get_op("Variance")()
std = get_op("Std")()
max = get_op("Max")()
min = get_op("Min")()
argmax = get_op("Argmax")()
argmin = get_op("Argmin")()
all = get_op("All")()
any = get_op("Any")()

logsumexp = get_op("Logsumexp")()
nanargmax = get_op("Nanargmax")()
nanargmin = get_op("Nanargmin")()
nancumprod = get_op("Nancumprod")()
nancumsum = get_op("Nancumsum")()
nanmax = get_op("Nanmax")()
nanmean = get_op("Nanmean")()
nanmedian = get_op("Nanmedian")()
nanmin = get_op("Nanmin")()
nanpercentile = get_op("Nanpercentile")()
nanprod = get_op("Nanprod")()
nanquantile = get_op("Nanquantile")()
nanstd = get_op("Nanstd")()
nansum = get_op("Nansum")()
nanvar = get_op("Nanvar")()
bincount = get_op("Bincount")()

count_nonzero = get_op("CountNonzero")()
norm = get_op("Norm")()
cumsum = get_op("Cumsum")()

__all__ = [
    "Pmean",
    "Psum",
    "ReduceWindow",
    "all",
    "any",
    "approx_max_k",
    "approx_min_k",
    "argmax",
    "argmin",
    "corrcoef",
    "correlate",
    "count_nonzero",
    "cov",
    "cumsum",
    "logsumexp",
    "max",
    "mean",
    "min",
    "moments",
    "norm",
    "pmean",
    "prod",
    "psum",
    "reduce_window",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "std",
    "sum",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "variance",
]
