"""Reductions."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, get_op, register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Mean")
class Mean(ReductionOp):
    """Mean reduction operation.

    Computes the arithmetic mean of elements across specified dimensions of an input
    tensor
    """

    op_name = "Mean"


@register_op("ApplyOverAxes")
class ApplyOverAxes(OpDef):
    """Apply a function repeatedly over multiple axes."""

    op_name = "ApplyOverAxes"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        # Typically shape does not change for some functions or reduces, fallback to None
        return None


@register_op("Bincount")
class Bincount(OpDef):
    """Bincount operation."""

    op_name = "Bincount"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        # Typically returns a 1D tensor whose size depends on max value, fallback to (None,)
        return (None,)


@register_op("Average")
class Average(ReductionOp):
    """Average reduction operation.

    Computes the weighted average along the specified axis
    """

    op_name = "Average"
    np_op_name = "average"


@register_op("Variance")
class Variance(ReductionOp):
    """Variance reduction operation.

    Computes the variance of elements across specified dimensions of an input
    tensor
    """

    op_name = "Variance"
    np_op_name = "var"


@register_op("Std")
class Std(ReductionOp):
    """Standard deviation reduction operation.

    Computes the standard deviation of elements across specified dimensions of an
    input tensor
    """

    op_name = "Std"
    np_op_name = "std"


@register_op("Corrcoef")
class Corrcoef(OpDef):
    """Return Pearson product-moment correlation coefficients."""

    op_name = "Corrcoef"
    np_op_name = "corrcoef"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None, None)


@register_op("Correlate")
class Correlate(OpDef):
    """Cross-correlation of two 1-dimensional sequences."""

    op_name = "Correlate"
    np_op_name = "correlate"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("Cov")
class Cov(OpDef):
    """Estimate a covariance matrix, given data and weights."""

    op_name = "Cov"
    np_op_name = "cov"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None, None)


@register_op("TrapezoidalIntegral")
class TrapezoidalIntegral(OpDef):
    """TrapezoidalIntegral operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        y = args[0] if len(args) > 0 else kwargs.get("y")
        shape = list(y)
        axis = kwargs.get("axis", -1)
        if axis < 0:
            axis += len(shape)
        shape.pop(axis)
        return tuple(shape)


@dispatch_eager("TrapezoidalIntegral")
def trapezoidal_integral(y: Tensor, x: Tensor = None, dx: float = 1.0, axis: int = -1) -> Tensor:
    """Compute the trapezoidal integral along a given axis."""
    return get_op("TrapezoidalIntegral")()(y, x=x, dx=dx, axis=axis)


@register_op("ConfusionMatrix")
class ConfusionMatrix(OpDef):
    """ConfusionMatrix operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        num_classes = kwargs.get("num_classes", 0)
        return (num_classes, num_classes)


@dispatch_eager("ConfusionMatrix")
def confusion_matrix(labels: Tensor, predictions: Tensor, num_classes: int, weights: Tensor = None) -> Tensor:
    """Compute confusion matrix."""
    return get_op("ConfusionMatrix")()(labels, predictions, num_classes=num_classes, weights=weights)


def moments(x: object, axes: object = None, keepdims: bool = False) -> tuple[object, object]:
    """Computes the mean and variance of x."""
    mean_op = get_op("Mean")()
    variance_op = get_op("Variance")()

    m = mean_op(x, axis=axes, keepdims=keepdims)
    v = variance_op(x, axis=axes, keepdims=keepdims)
    return m, v


@register_op("Descriptive")
class Descriptive(OpDef):
    """Descriptive operation."""

    op_name = "Descriptive"


@dispatch_eager("Descriptive")
def descriptive(a: object) -> object:
    """Function for descriptive."""
    return get_op("Descriptive")()(a)


@register_op("Distributions")
class Distributions(OpDef):
    """Distributions operation."""

    op_name = "Distributions"


@dispatch_eager("Distributions")
def distributions(a: object) -> object:
    """Function for distributions."""
    return get_op("Distributions")()(a)
