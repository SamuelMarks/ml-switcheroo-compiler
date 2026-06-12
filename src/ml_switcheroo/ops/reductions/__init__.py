"""Reduction operations package."""

from ml_switcheroo.ops.reductions.basic import (
    ReductionOp,
    Sum,
    Mean,
    Max,
    Min,
)

__all__ = [
    "ReductionOp",
    "Sum",
    "Mean",
    "Max",
    "Min",
]
from .frontend import (
    all as all,
    any as any,
    argmax as argmax,
    argmin as argmin,
    count_nonzero as count_nonzero,
    cumsum as cumsum,
    logsumexp as logsumexp,
    max as max,
    mean as mean,
    min as min,
    norm as norm,
    prod as prod,
    std as std,
    sum as sum,
    variance as variance,
)
