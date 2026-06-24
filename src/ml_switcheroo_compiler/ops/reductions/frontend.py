"""Reduction operations frontend."""

from .frontend_utils import reduce_window
from .frontend_segment import (
    segment_sum,
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
)
from .frontend_pool import (
    fractional_max_pool2d,
    adaptive_avg_pool2d,
    adaptive_max_pool2d,
    unfold,
    fold,
)
from .frontend_stats import (
    psum,
    pmean,
    ctc_loss,
    approx_max_k,
    approx_min_k,
    corrcoef,
    correlate,
    cov,
)

__all__ = [
    "reduce_window",
    "segment_sum",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "fractional_max_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "unfold",
    "fold",
    "psum",
    "pmean",
    "ctc_loss",
    "approx_max_k",
    "approx_min_k",
    "corrcoef",
    "correlate",
    "cov",
]
