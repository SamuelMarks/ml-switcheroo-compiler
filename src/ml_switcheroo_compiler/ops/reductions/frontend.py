"""Reduction operations frontend."""

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

__all__ = [
    "adaptive_avg_pool2d",
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
