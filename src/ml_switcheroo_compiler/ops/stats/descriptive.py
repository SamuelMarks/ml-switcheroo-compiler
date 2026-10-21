"""Descriptive statistics, moments, covariance, and correlation operations."""

from __future__ import annotations

from collections.abc import Iterable

from ml_switcheroo_compiler.backends.registry import as_eager_list, as_int
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, get_op, register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Mean")
class Mean(ReductionOp):
    """Arithmetic mean reduction operation over specified tensor dimensions."""

    op_name = "Mean"


@register_op("ApplyOverAxes")
class ApplyOverAxes(OpDef):
    """Apply a reduction function repeatedly over multiple axes."""

    op_name = "ApplyOverAxes"

    def infer_shape(
        self,
        func: TensorOrScalar = None,
        a: TensorOrScalar = None,
        axes: TensorOrScalar = None,
        **kwargs: TensorOrScalar,
    ) -> tuple[int, ...]:
        """Infer shape after reducing specified axes to size 1.

        Args:
            func (TensorOrScalar): Applied reduction callable.
            a (TensorOrScalar): Input tensor.
            axes (TensorOrScalar): Sequence of axes to reduce.
            **kwargs (TensorOrScalar): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Axis-reduced shape with reduced dimensions set to 1.
        """
        inp_a = (
            a
            if a is not None
            else kwargs.get(
                "a",
                kwargs.get(
                    "x",
                    func if not callable(func) and (hasattr(func, "shape") or isinstance(func, (list, tuple))) else None,
                ),
            )
        )
        if inp_a is None:
            return ()

        def _get_shape(obj: TensorOrScalar) -> tuple[int, ...]:
            """Extract integer shape tuple from a tensor or sequence.

            Args:
                obj (TensorOrScalar): Target tensor or shape sequence.

            Returns:
                tuple[int, ...]: Extracted shape tuple.
            """
            if hasattr(obj, "shape"):
                raw_shape = getattr(obj, "shape")
                return tuple(int(d) for d in raw_shape) if isinstance(raw_shape, Iterable) else ()
            if hasattr(obj, "shape_metadata"):
                sm = getattr(obj, "shape_metadata")
                if sm:
                    return tuple(int(d) for d in sm)
            if isinstance(obj, (list, tuple)):
                return tuple(int(d) for d in obj)
            return ()

        shape = list(_get_shape(inp_a))
        ax = axes if axes is not None else kwargs.get("axes")
        if ax is not None and shape:
            axes_list = [int(ax)] if isinstance(ax, int) else [int(as_int(x)) for x in (as_eager_list(ax) if isinstance(ax, Iterable) else ())]
            for a_idx in axes_list:
                norm_idx = a_idx + len(shape) if a_idx < 0 else a_idx
                if 0 <= norm_idx < len(shape):
                    shape[norm_idx] = 1
        return tuple(shape)


@register_op("Bincount")
class Bincount(OpDef):
    """Count number of occurrences of each non-negative integer value."""

    op_name = "Bincount"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer dynamic 1D output shape for Bincount.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Dynamic 1-tuple `(None,)`.
        """
        _ = (args, kwargs)
        return (None,)


@register_op("Average")
class Average(ReductionOp):
    """Weighted average reduction operation along a specified axis."""

    op_name = "Average"
    np_op_name = "average"


@register_op("Variance")
class Variance(ReductionOp):
    """Variance reduction operation across specified tensor dimensions."""

    op_name = "Variance"
    np_op_name = "var"


@register_op("Std")
class Std(ReductionOp):
    """Standard deviation reduction operation across specified tensor dimensions."""

    op_name = "Std"
    np_op_name = "std"


@register_op("Corrcoef")
class Corrcoef(OpDef):
    """Compute Pearson product-moment correlation coefficient matrix."""

    op_name = "Corrcoef"
    np_op_name = "corrcoef"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 2D output shape for Corrcoef.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Dynamic 2D shape tuple `(None, None)`.
        """
        _ = (args, kwargs)
        return (None, None)


@register_op("Correlate")
class Correlate(OpDef):
    """Cross-correlation of two 1-dimensional sequences."""

    op_name = "Correlate"
    np_op_name = "correlate"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D output shape for Correlate.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Dynamic 1D shape tuple `(None,)`.
        """
        _ = (args, kwargs)
        return (None,)


@register_op("Cov")
class Cov(OpDef):
    """Estimate a covariance matrix from data and optional weights."""

    op_name = "Cov"
    np_op_name = "cov"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 2D output shape for Cov.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Dynamic 2D shape tuple `(None, None)`.
        """
        _ = (args, kwargs)
        return (None, None)


@register_op("TrapezoidalIntegral")
class TrapezoidalIntegral(OpDef):
    """Integrate along the given axis using the composite trapezoidal rule."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape after integrating along `axis`.

        Args:
            *args (TensorOrScalar): Positional input shape tuple `(y_shape, ...)`.
            **kwargs (TensorOrScalar): Keyword attributes including `axis`.

        Returns:
            TensorOrScalar: Reduced output shape tuple.
        """
        y = args[0] if len(args) > 0 else kwargs.get("y")
        shape = list(y) if isinstance(y, Iterable) else []
        axis = int(as_int(kwargs.get("axis", -1)))
        if axis < 0:
            axis += len(shape)
        shape.pop(axis)
        return tuple(shape)


@dispatch_eager("TrapezoidalIntegral")
def trapezoidal_integral(y: Tensor, x: TensorOrScalar = None, dx: TensorOrScalar = 1.0, axis: TensorOrScalar = -1) -> TensorOrScalar:
    """Integrate `y` along `axis` using the composite trapezoidal rule.

    Args:
        y (Tensor): Input tensor to integrate.
        x (TensorOrScalar): Optional sample point coordinates corresponding to `y`.
        dx (TensorOrScalar): Spacing between sample points when `x` is None.
        axis (TensorOrScalar): Axis along which to integrate.

    Returns:
        TensorOrScalar: Definite integral approximated by the trapezoidal rule.
    """
    return get_op("TrapezoidalIntegral")()(y, x=x, dx=dx, axis=axis)


@register_op("ConfusionMatrix")
class ConfusionMatrix(OpDef):
    """Compute the confusion matrix from predictions and ground-truth labels."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer `(num_classes, num_classes)` output shape for ConfusionMatrix.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword attributes including `num_classes`.

        Returns:
            TensorOrScalar: 2D shape tuple `(num_classes, num_classes)`.
        """
        _ = args
        num_classes = kwargs.get("num_classes", 0)
        return (num_classes, num_classes)


@dispatch_eager("ConfusionMatrix")
def confusion_matrix(
    labels: Tensor,
    predictions: Tensor,
    num_classes: int,
    weights: TensorOrScalar = None,
) -> TensorOrScalar:
    """Compute a `(num_classes, num_classes)` confusion matrix for classification evaluation.

    Args:
        labels (Tensor): 1D tensor of ground-truth class labels.
        predictions (Tensor): 1D tensor of predicted class labels.
        num_classes (int): Total number of classification categories.
        weights (TensorOrScalar): Optional sample weight tensor.

    Returns:
        TensorOrScalar: Confusion matrix tensor of shape `(num_classes, num_classes)`.
    """
    return get_op("ConfusionMatrix")()(labels, predictions, num_classes=num_classes, weights=weights)


def moments(x: TensorOrScalar, axes: TensorOrScalar = None, keepdims: bool = False) -> TensorOrScalar:
    """Calculate the mean and variance of `x` across `axes`.

    Args:
        x (TensorOrScalar): Input tensor.
        axes (TensorOrScalar): Axis or sequence of axes along which to compute moments.
        keepdims (bool): Whether to retain reduced dimensions with length 1.

    Returns:
        TensorOrScalar: Tuple of `(mean_tensor, variance_tensor)`.
    """
    mean_op = get_op("Mean")()
    variance_op = get_op("Variance")()

    m = mean_op(x, axis=axes, keepdims=keepdims)
    v = variance_op(x, axis=axes, keepdims=keepdims)
    return m, v


@register_op("Descriptive")
class Descriptive(OpDef):
    """Compute descriptive summary statistics (mean, variance, skewness, kurtosis)."""

    op_name = "Descriptive"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 4-tuple of output shapes for Descriptive statistics.

        Args:
            *args (TensorOrScalar): Positional input tensor arguments.
            **kwargs (TensorOrScalar): Keyword attributes (`axis`, `keepdims`).

        Returns:
            TensorOrScalar: 4-tuple of reduced shape tuples.
        """
        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)

        if axis is None:
            out_shape = tuple(1 for _ in shape) if keepdims else ()
            return (out_shape, out_shape, out_shape, out_shape)

        axes_seq = (axis,) if isinstance(axis, int) else (tuple(axis) if isinstance(axis, Iterable) else ())
        if keepdims:
            out_shape_list = list(shape)
            for d in axes_seq:
                out_shape_list[int(d)] = 1
            out_shape = tuple(out_shape_list)
        else:
            dim_indices = [int(d) if int(d) >= 0 else len(shape) + int(d) for d in axes_seq]
            out_shape = tuple(s for i, s in enumerate(shape) if i not in dim_indices)

        return (out_shape, out_shape, out_shape, out_shape)


@dispatch_eager("Descriptive")
def descriptive(a: TensorOrScalar) -> TensorOrScalar:
    """Compute descriptive statistics for input tensor `a`.

    Args:
        a (TensorOrScalar): Input tensor.

    Returns:
        TensorOrScalar: Descriptive summary statistics result.
    """
    return get_op("Descriptive")()(a)


@register_op("Distributions")
class Distributions(OpDef):
    """Probability distribution sampling and evaluation operation."""

    op_name = "Distributions"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape combining `sample_shape` with input batch/event shape.

        Args:
            *args (TensorOrScalar): Positional input arguments.
            **kwargs (TensorOrScalar): Keyword attributes including `sample_shape`.

        Returns:
            TensorOrScalar: Concatenated `(sample_shape + event_shape)` tuple.
        """
        a = args[0]
        shape = tuple(int(d) for d in getattr(a, "shape", getattr(a, "shape_metadata", ())))
        sample_shape = kwargs.get("sample_shape", ())
        if isinstance(sample_shape, int):
            sample_shape_tuple: tuple[int, ...] = (sample_shape,)
        else:
            sample_shape_tuple = tuple(int(as_int(d)) for d in as_eager_list(sample_shape or ()))
        return sample_shape_tuple + shape


@dispatch_eager("Distributions")
def distributions(a: TensorOrScalar) -> TensorOrScalar:
    """Evaluate probability distribution operation on `a`.

    Args:
        a (TensorOrScalar): Input distribution parameter tensor.

    Returns:
        TensorOrScalar: Distribution evaluation result.
    """
    return get_op("Distributions")()(a)
