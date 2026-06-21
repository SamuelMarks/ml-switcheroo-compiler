"""NaN-safe reduction operations."""

from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Nanargmax")
class Nanargmax(ReductionOp):
    """Computes the index of the maximum value, ignoring NaNs."""

    op_name = "Nanargmax"
    np_op_name = "nanargmax"


@register_op("Nanargmin")
class Nanargmin(ReductionOp):
    """Computes the index of the minimum value, ignoring NaNs."""

    op_name = "Nanargmin"
    np_op_name = "nanargmin"


@register_op("Nancumprod")
class Nancumprod(ReductionOp):
    """Computes the cumulative product, ignoring NaNs."""

    op_name = "Nancumprod"
    np_op_name = "nancumprod"


@register_op("Nancumsum")
class Nancumsum(ReductionOp):
    """Computes the cumulative sum, ignoring NaNs."""

    op_name = "Nancumsum"
    np_op_name = "nancumsum"


@register_op("Nanmax")
class Nanmax(ReductionOp):
    """Computes the maximum value, ignoring NaNs."""

    op_name = "Nanmax"
    np_op_name = "nanmax"


@register_op("Nanmean")
class Nanmean(ReductionOp):
    """Computes the mean, ignoring NaNs."""

    op_name = "Nanmean"
    np_op_name = "nanmean"


@register_op("Nanmedian")
class Nanmedian(ReductionOp):
    """Computes the median, ignoring NaNs."""

    op_name = "Nanmedian"
    np_op_name = "nanmedian"


@register_op("Nanmin")
class Nanmin(ReductionOp):
    """Computes the minimum value, ignoring NaNs."""

    op_name = "Nanmin"
    np_op_name = "nanmin"


@register_op("Nanpercentile")
class Nanpercentile(ReductionOp):
    """Computes the percentile, ignoring NaNs."""

    op_name = "Nanpercentile"
    np_op_name = "nanpercentile"


@register_op("Nanprod")
class Nanprod(ReductionOp):
    """Computes the product, ignoring NaNs."""

    op_name = "Nanprod"
    np_op_name = "nanprod"


@register_op("Nanquantile")
class Nanquantile(ReductionOp):
    """Computes the quantile, ignoring NaNs."""

    op_name = "Nanquantile"
    np_op_name = "nanquantile"


@register_op("Nanstd")
class Nanstd(ReductionOp):
    """Computes the standard deviation, ignoring NaNs."""

    op_name = "Nanstd"
    np_op_name = "nanstd"


@register_op("Nansum")
class Nansum(ReductionOp):
    """Computes the sum, ignoring NaNs."""

    op_name = "Nansum"
    np_op_name = "nansum"


@register_op("Nanvar")
class Nanvar(ReductionOp):
    """Computes the variance, ignoring NaNs."""

    op_name = "Nanvar"
    np_op_name = "nanvar"
