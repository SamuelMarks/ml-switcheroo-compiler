# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module nan.py."""

"""NaN-safe reduction operations."""

from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Nanargmax")
class Nanargmax(ReductionOp):
    """Compute the index of the maximum value, ignoring NaNs."""

    op_name: object = "Nanargmax"
    np_op_name: object = "nanargmax"


@register_op("Nanargmin")
class Nanargmin(ReductionOp):
    """Compute the index of the minimum value, ignoring NaNs."""

    op_name: object = "Nanargmin"
    np_op_name: object = "nanargmin"


@register_op("Nancumprod")
class Nancumprod(ReductionOp):
    """Compute the cumulative product, ignoring NaNs."""

    op_name: object = "Nancumprod"
    np_op_name: object = "nancumprod"


@register_op("Nancumsum")
class Nancumsum(ReductionOp):
    """Compute the cumulative sum, ignoring NaNs."""

    op_name: object = "Nancumsum"
    np_op_name: object = "nancumsum"


@register_op("Nanmax")
class Nanmax(ReductionOp):
    """Compute the maximum value, ignoring NaNs."""

    op_name: object = "Nanmax"
    np_op_name: object = "nanmax"


@register_op("Nanmean")
class Nanmean(ReductionOp):
    """Compute the mean, ignoring NaNs."""

    op_name: object = "Nanmean"
    np_op_name: object = "nanmean"


@register_op("Nanmedian")
class Nanmedian(ReductionOp):
    """Compute the median, ignoring NaNs."""

    op_name: object = "Nanmedian"
    np_op_name: object = "nanmedian"


@register_op("Nanmin")
class Nanmin(ReductionOp):
    """Compute the minimum value, ignoring NaNs."""

    op_name: object = "Nanmin"
    np_op_name: object = "nanmin"


@register_op("Nanpercentile")
class Nanpercentile(ReductionOp):
    """Compute the percentile, ignoring NaNs."""

    op_name: object = "Nanpercentile"
    np_op_name: object = "nanpercentile"


@register_op("Nanprod")
class Nanprod(ReductionOp):
    """Compute the product, ignoring NaNs."""

    op_name: object = "Nanprod"
    np_op_name: object = "nanprod"


@register_op("Nanquantile")
class Nanquantile(ReductionOp):
    """Compute the quantile, ignoring NaNs."""

    op_name: object = "Nanquantile"
    np_op_name: object = "nanquantile"


@register_op("Nanstd")
class Nanstd(ReductionOp):
    """Compute the standard deviation, ignoring NaNs."""

    op_name: object = "Nanstd"
    np_op_name: object = "nanstd"


@register_op("Nansum")
class Nansum(ReductionOp):
    """Compute the sum, ignoring NaNs."""

    op_name: object = "Nansum"
    np_op_name: object = "nansum"


@register_op("Nanvar")
class Nanvar(ReductionOp):
    """Compute the variance, ignoring NaNs."""

    op_name: object = "Nanvar"
    np_op_name: object = "nanvar"
