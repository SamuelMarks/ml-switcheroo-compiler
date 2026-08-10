from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Reductions."""
from typing import Any

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

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        # Typically shape does not change for some functions or reduces, fallback to None
        return None


@register_op("Bincount")
class Bincount(OpDef):
    """Bincount operation."""

    op_name = "Bincount"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
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

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None, None)


@register_op("Correlate")
class Correlate(OpDef):
    """Cross-correlation of two 1-dimensional sequences."""

    op_name = "Correlate"
    np_op_name = "correlate"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None,)


@register_op("Cov")
class Cov(OpDef):
    """Estimate a covariance matrix, given data and weights."""

    op_name = "Cov"
    np_op_name = "cov"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None, None)


@register_op("TrapezoidalIntegral")
class TrapezoidalIntegral(OpDef):
    """TrapezoidalIntegral operation."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        y = args[0] if len(args) > 0 else kwargs.get("y")
        shape = list(y)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        axis = kwargs.get("axis", -1)
        if axis < 0:
            axis += len(shape)
        shape.pop(axis)
        return tuple(shape)


@dispatch_eager("TrapezoidalIntegral")
def trapezoidal_integral(y: Tensor, x: Any = None, dx: Any = 1.0, axis: Any = -1) -> Any:
    """Evaluate trapezoidal_integral operation.

    Args:
        y (Tensor): The y parameter.
        x (Tensor): The x parameter.
        dx (float): The dx parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    return get_op("TrapezoidalIntegral")()(y, x=x, dx=dx, axis=axis)


@register_op("ConfusionMatrix")
class ConfusionMatrix(OpDef):
    """ConfusionMatrix operation."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        num_classes = kwargs.get("num_classes", 0)
        return (num_classes, num_classes)


@dispatch_eager("ConfusionMatrix")
def confusion_matrix(labels: Tensor, predictions: Tensor, num_classes: int, weights: Any = None) -> Any:
    """Evaluate confusion_matrix operation.

    Args:
        labels (Tensor): The labels parameter.
        predictions (Tensor): The predictions parameter.
        num_classes (int): The num_classes parameter.
        weights (Tensor): The weights parameter.

    Returns:
        Tensor: Result.
    """
    return get_op("ConfusionMatrix")()(labels, predictions, num_classes=num_classes, weights=weights)


def moments(x: Any, axes: Any = None, keepdims: bool = False) -> tuple[Any, Any]:
    """Compute the mean and variance of x.

    Args:
        x (object): The x parameter.
        axes (object): The axes parameter.
        keepdims (bool): The keepdims parameter.

    Returns: Any: Result.
    """
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
def descriptive(a: Any) -> Any:
    """Provide function for descriptive.

    Args:
        a (object): The a parameter.

    Returns: Any: Result.
    """
    return get_op("Descriptive")()(a)


@register_op("Distributions")
class Distributions(OpDef):
    """Distributions operation."""

    op_name = "Distributions"


@dispatch_eager("Distributions")
def distributions(a: Any) -> Any:
    """Provide function for distributions.

    Args:
        a (object): The a parameter.

    Returns: Any: Result.
    """
    return get_op("Distributions")()(a)
