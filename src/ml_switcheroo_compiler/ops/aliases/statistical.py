"""Aliases for statistical."""

from ml_switcheroo_compiler.ops.reductions import max, min, variance

from .common import create_eager_alias

amin = min

amax = max

var = variance


def average(
    a: object, axis: int = None, weights: object = None, returned: bool = False, **kwargs: object
) -> object:
    """Compute the weighted average along the specified axis."""
    from ml_switcheroo_compiler.ops.binary import multiply
    from ml_switcheroo_compiler.ops.creation.frontend import asarray
    from ml_switcheroo_compiler.ops.reductions import mean
    from ml_switcheroo_compiler.ops.reductions import sum as sum_op

    a = asarray(a)
    if weights is None:
        avg = mean(a, axis=axis, **kwargs)
        return (avg, sum_op(asarray(1.0))) if returned else avg
    weights = asarray(weights)
    w_sum = sum_op(weights, axis=axis, **kwargs)
    avg = sum_op(multiply(a, weights), axis=axis, **kwargs) / w_sum
    return (avg, w_sum) if returned else avg


histogram = create_eager_alias("histogram")


histogram2d = create_eager_alias("histogram2d")


histogram_bin_edges = create_eager_alias("histogram_bin_edges")


histogramdd = create_eager_alias("histogramdd")


median = create_eager_alias("median")


percentile = create_eager_alias("percentile")


ptp = create_eager_alias("ptp")


quantile = create_eager_alias("quantile")
