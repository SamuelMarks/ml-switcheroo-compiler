"""Module descriptive.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Reductions."""

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

    def infer_shape(self, func=None, a=None, axes=None, **kwargs) -> tuple[int, ...]:
        """Infer shape after reducing specified axes to size 1.

        Args:
            func (object): Applied function.
            a (object): Input tensor.
            axes (object): Sequence of axes to reduce.
            **kwargs (object): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Axis-reduced shape with reduced dimensions set to 1.
        """
        inp_a = a if a is not None else kwargs.get("a", kwargs.get("x", func if not callable(func) and (hasattr(func, "shape") or isinstance(func, (list, tuple))) else None))
        if inp_a is None:
            return ()

        def _get_shape(obj: object) -> tuple[int, ...]:
            """Extract shape tuple from object.

            Args:
                obj (object): Target tensor or shape.

            Returns:
                tuple[int, ...]: Extracted shape tuple.
            """
            if hasattr(obj, "shape"):
                return tuple(int(d) for d in obj.shape)
            if hasattr(obj, "shape_metadata"):
                sm = obj.shape_metadata
                if sm:
                    return tuple(int(d) for d in sm)
            if isinstance(obj, (list, tuple)):
                return tuple(int(d) for d in obj)
            return ()

        shape = list(_get_shape(inp_a))
        ax = axes if axes is not None else kwargs.get("axes")
        if ax is not None and shape:
            axes_list = [int(ax)] if isinstance(ax, int) else [int(x) for x in ax]
            for a_idx in axes_list:
                norm_idx = a_idx + len(shape) if a_idx < 0 else a_idx
                if 0 <= norm_idx < len(shape):
                    shape[norm_idx] = 1
        return tuple(shape)


@register_op("Bincount")
class Bincount(OpDef):
    """Bincount operation."""

    op_name = "Bincount"

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None, None)


@register_op("Correlate")
class Correlate(OpDef):
    """Cross-correlation of two 1-dimensional sequences."""

    op_name = "Correlate"
    np_op_name = "correlate"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None,)


@register_op("Cov")
class Cov(OpDef):
    """Estimate a covariance matrix, given data and weights."""

    op_name = "Cov"
    np_op_name = "cov"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (None, None)


@register_op("TrapezoidalIntegral")
class TrapezoidalIntegral(OpDef):
    """TrapezoidalIntegral operation."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        y = args[0] if len(args) > 0 else kwargs.get("y")
        shape = list(y)
        axis = kwargs.get("axis", -1)
        if axis < 0:
            axis += len(shape)
        shape.pop(axis)
        return tuple(shape)


@dispatch_eager("TrapezoidalIntegral")
def trapezoidal_integral(y: Tensor, x=None, dx=1.0, axis=-1):
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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        num_classes = kwargs.get("num_classes", 0)
        return (num_classes, num_classes)


@dispatch_eager("ConfusionMatrix")
def confusion_matrix(labels: Tensor, predictions: Tensor, num_classes: int, weights=None):
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


def moments(x, axes=None, keepdims: bool = False):
    """Compute the mean and variance of x.

    Args:
        x (Any): The x parameter.
        axes (Any): The axes parameter.
        keepdims (bool): The keepdims parameter.

    Returns:
            tuple[int, ...]: Result.
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
def descriptive(a):
    """Provide function for descriptive.

    Args:
        a (Any): The a parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return get_op("Descriptive")()(a)


@register_op("Distributions")
class Distributions(OpDef):
    """Distributions operation."""

    op_name = "Distributions"


@dispatch_eager("Distributions")
def distributions(a):
    """Provide function for distributions.

    Args:
        a (Any): The a parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return get_op("Distributions")()(a)
