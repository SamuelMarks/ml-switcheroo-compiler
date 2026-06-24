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
    from ml_switcheroo_compiler.ops.binary import multiply  # pragma: no cover
    from ml_switcheroo_compiler.ops.creation.frontend import asarray  # pragma: no cover
    from ml_switcheroo_compiler.ops.reductions import mean  # pragma: no cover
    from ml_switcheroo_compiler.ops.reductions import sum as sum_op  # pragma: no cover

    a = asarray(a)  # pragma: no cover
    if weights is None:  # pragma: no cover
        avg = mean(a, axis=axis, **kwargs)  # pragma: no cover
        return (avg, sum_op(asarray(1.0))) if returned else avg  # pragma: no cover
    weights = asarray(weights)  # pragma: no cover
    w_sum = sum_op(weights, axis=axis, **kwargs)  # pragma: no cover
    avg = sum_op(multiply(a, weights), axis=axis, **kwargs) / w_sum  # pragma: no cover
    return (avg, w_sum) if returned else avg  # pragma: no cover


histogram = create_eager_alias("histogram")


histogram2d = create_eager_alias("histogram2d")


histogram_bin_edges = create_eager_alias("histogram_bin_edges")


histogramdd = create_eager_alias("histogramdd")


median = create_eager_alias("median")


percentile = create_eager_alias("percentile")


ptp = create_eager_alias("ptp")


quantile = create_eager_alias("quantile")
