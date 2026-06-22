"""Reductions."""

from __future__ import annotations


from ml_switcheroo_compiler.ops.base import OpDef, register_op, dispatch_eager, get_op
from ml_switcheroo_compiler.core.tensor import Tensor


from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Mean")
class Mean(ReductionOp):
    """Mean reduction operation.

    Computes the arithmetic mean of elements across specified dimensions of an input
    tensor
    """

    op_name = "Mean"


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

    def infer_shape(
        self, x: object, y: object = None, rowvar: bool = True, **kwargs: object
    ) -> object:
        """Infer the output shape."""
        return (None, None)


@register_op("Correlate")
class Correlate(OpDef):
    """Cross-correlation of two 1-dimensional sequences."""

    op_name = "Correlate"
    np_op_name = "correlate"

    def infer_shape(self, a: object, v: object, mode: str = "valid", **kwargs: object) -> object:
        """Infer the output shape."""
        return (None,)


@register_op("Cov")
class Cov(OpDef):
    """Estimate a covariance matrix, given data and weights."""

    op_name = "Cov"
    np_op_name = "cov"

    def infer_shape(
        self, m: object, y: object = None, rowvar: bool = True, **kwargs: object
    ) -> object:
        """Infer the output shape."""
        return (None, None)


@register_op("TrapezoidalIntegral")
class TrapezoidalIntegral(OpDef):
    """TrapezoidalIntegral operation."""

    def infer_shape(self, y: object, **kwargs: object) -> object:
        """Infer shape."""
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

    def infer_shape(self, labels: object, predictions: object, **kwargs: object) -> object:
        """Infer shape."""
        num_classes = kwargs.get("num_classes", 0)
        return (num_classes, num_classes)


@dispatch_eager("ConfusionMatrix")
def confusion_matrix(
    labels: Tensor, predictions: Tensor, num_classes: int, weights: Tensor = None
) -> Tensor:
    """Compute confusion matrix."""
    return get_op("ConfusionMatrix")()(
        labels, predictions, num_classes=num_classes, weights=weights
    )
