# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Misc operations."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Finfo")
class Finfo(OpDef):
    """Finfo operation."""

    op_name = "Finfo"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("Iinfo")
class Iinfo(OpDef):
    """Iinfo operation."""

    op_name = "Iinfo"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("GetPrintoptions")
class GetPrintoptions(OpDef):
    """Get the current print options."""

    op_name = "GetPrintoptions"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("Gradient")
class Gradient(OpDef):
    """Return the gradient of an N-dimensional array."""

    op_name = "Gradient"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Histogram")
class Histogram(OpDef):
    """Compute the histogram of a dataset."""

    op_name = "Histogram"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        bins = kwargs.get("bins", 10)
        if hasattr(bins, "shape") and len(bins.shape) > 0:
            return (bins.shape[0] - 1,)
        if isinstance(bins, int):
            return (bins,)
        return (10,)


@register_op("Histogram2d")
class Histogram2d(OpDef):
    """Compute the bi-dimensional histogram of two data samples."""

    op_name = "Histogram2d"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        bins = kwargs.get("bins", 10)
        if isinstance(bins, (list, tuple)):
            if len(bins) == 2:
                b1, b2 = bins
                b1_len = b1 if isinstance(b1, int) else (b1.shape[0] - 1 if hasattr(b1, "shape") else 10)
                b2_len = b2 if isinstance(b2, int) else (b2.shape[0] - 1 if hasattr(b2, "shape") else 10)
                return (b1_len, b2_len)
        return (10, 10)


@register_op("HistogramBinEdges")
class HistogramBinEdges(OpDef):
    """Provide function to calculate only the edges of the bins used by the histogram function."""

    op_name = "HistogramBinEdges"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        bins = kwargs.get("bins", 10)
        if hasattr(bins, "shape") and len(bins.shape) > 0:
            return bins.shape
        if isinstance(bins, int):
            return (bins + 1,)
        return (11,)


@register_op("Histogramdd")
class Histogramdd(OpDef):
    """Compute the multidimensional histogram of some data."""

    op_name = "Histogramdd"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        sample = args[0]
        n_dim = sample.shape[1] if (hasattr(sample, "shape") and len(sample.shape) == 2) else 1
        return tuple(10 for _ in range(n_dim))


@register_op("I0")
class I0(OpDef):
    """Modify Bessel function of the first kind, order 0."""

    op_name = "I0"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Indices")
class Indices(OpDef):
    """Return an array representing the indices of a grid."""

    op_name = "Indices"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        dimensions = args[0]
        dim_len = len(dimensions) if isinstance(dimensions, (list, tuple)) else 0
        return (dim_len, *dimensions) if dim_len > 0 else ()


@register_op("Infeed")
class Infeed(OpDef):
    """Read from the infeed queue."""

    op_name = "Infeed"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return kwargs.get("shape", ())


@register_op("Interp")
class Interp(OpDef):
    """One-dimensional linear interpolation."""

    op_name = "Interp"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Intersect1d")
class Intersect1d(OpDef):
    """Find the intersection of two arrays."""

    op_name = "Intersect1d"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None,)


@register_op("Isscalar")
class Isscalar(OpDef):
    """Return True if the type of num is a scalar type."""

    op_name = "Isscalar"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("Iterable")
class Iterable(OpDef):
    """Check whether or not an object can be iterated over."""

    op_name = "Iterable"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("Ix")
class Ix(OpDef):
    """Construct an open mesh from multiple sequences."""

    op_name = "Ix"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        nd = len(args)
        if nd == 0:
            return ()
        # ix_ returns a tuple of ndarrays, each having ndim == nd. We return shape of the first output.
        # Actually it returns a tuple of arrays, the Op should maybe return a tuple of shapes,
        # but since we can only return one shape, let's return the shape of the first one.
        shape = [1] * nd
        if hasattr(args[0], "shape") and len(args[0].shape) > 0:
            shape[0] = args[0].shape[0]
        return tuple(shape)


@register_op("Kron")
class Kron(OpDef):
    """Kronecker product of two arrays."""

    op_name = "Kron"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if len(args) < 2:
            return ()
        a, b = args[0], args[1]
        shape_a = getattr(a, "shape", ())
        shape_b = getattr(b, "shape", ())
        ndims = max(len(shape_a), len(shape_b))
        shape_a = (1,) * (ndims - len(shape_a)) + shape_a
        shape_b = (1,) * (ndims - len(shape_b)) + shape_b
        return tuple(a_dim * b_dim for a_dim, b_dim in zip(shape_a, shape_b))


@register_op("MaskIndices")
class MaskIndices(OpDef):
    """Return the indices to access (n, n) arrays."""

    op_name = "MaskIndices"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None,)


@register_op("Median")
class Median(OpDef):
    """Compute the median along the specified axis."""

    op_name = "Median"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        shape = list(getattr(args[0], "shape", ()))
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        if axis is None:
            return (1,) if keepdims else ()
        if isinstance(axis, int):
            axis = [axis]
        for ax in sorted(axis, reverse=True):
            if ax < len(shape):
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
        return tuple(shape)


@register_op("Mgrid")
class Mgrid(OpDef):
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'."""

    op_name = "Mgrid"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return kwargs.get("shape", ())


@register_op("Mish")
class Mish(OpDef):
    """Mish activation function."""

    op_name = "Mish"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Modf")
class Modf(OpDef):
    """Return the fractional and integral parts of an array, element-wise."""

    op_name = "Modf"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Ogrid")
class Ogrid(OpDef):
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'."""

    op_name = "Ogrid"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return kwargs.get("shape", ())


@register_op("Piecewise")
class Piecewise(OpDef):
    """Evaluate a piecewise-defined function."""

    op_name = "Piecewise"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("PromoteTypes")
class PromoteTypes(OpDef):
    """Return the data type with the smallest size and smallest scalar kind."""

    op_name = "PromoteTypes"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("R")
class R(OpDef):
    """Translate slice objects to concatenation along the first axis."""

    op_name = "R"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None,)


@register_op("ResultType")
class ResultType(OpDef):
    """Return the type that results from applying the NumPy type promotion rules."""

    op_name = "ResultType"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("Rot90")
class Rot90(OpDef):
    """Rotate an array by 90 degrees in the plane specified by axes."""

    op_name = "Rot90"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        shape = list(getattr(args[0], "shape", ()))
        axes = kwargs.get("axes", (0, 1))
        if len(axes) == 2 and axes[0] < len(shape) and axes[1] < len(shape):
            shape[axes[0]], shape[axes[1]] = shape[axes[1]], shape[axes[0]]
        return tuple(shape)


@register_op("Trapezoid")
class Trapezoid(OpDef):
    """Integrate along the given axis using the composite trapezoidal rule."""

    op_name = "Trapezoid"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        shape = list(getattr(args[0], "shape", ()))
        axis = kwargs.get("axis", -1)
        if axis < len(shape):
            shape.pop(axis)
        return tuple(shape)


@register_op("Tri")
class Tri(OpDef):
    """Provide an array with ones at and below the given diagonal and zeros elsewhere."""

    op_name = "Tri"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        N = args[0] if args else 0
        M = kwargs.get("M", N)
        return (N, M)


@register_op("Tril")
class Tril(OpDef):
    """Lower triangle of an array."""

    op_name = "Tril"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("TrimZeros")
class TrimZeros(OpDef):
    """Trim the leading and/or trailing zeros from a 1-D array or sequence."""

    op_name = "TrimZeros"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (None,)


@register_op("Triu")
class Triu(OpDef):
    """Upper triangle of an array."""

    op_name = "Triu"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Unwrap")
class Unwrap(OpDef):
    """Unwrap by taking the complement of large deltas with respect to the period."""

    op_name = "Unwrap"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


@register_op("Vander")
class Vander(OpDef):
    """Generate a Vandermonde matrix."""

    op_name = "Vander"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        x = getattr(args[0], "shape", ())
        N = kwargs.get("N", x[0] if x else 0)
        return (*x, N)


@register_op("Vectorize")
class Vectorize(OpDef):
    """Generalized function class."""

    op_name = "Vectorize"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("AxisIndex")
class AxisIndex(OpDef):
    """AxisIndex operation."""

    op_name = "AxisIndex"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


def mgrid(*args: Any, **kwargs: Any) -> Any:
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mgrid", *args, **kwargs)


def ogrid(*args: Any, **kwargs: Any) -> Any:
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ogrid", *args, **kwargs)


def r_(*args: Any, **kwargs: Any) -> Any:
    """Translate slice objects to concatenation along the first axis.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("R", *args, **kwargs)


def gradient(*args: Any, **kwargs: Any) -> Any:
    """Return the gradient of an N-dimensional array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Gradient", *args, **kwargs)


def histogram(*args: Any, **kwargs: Any) -> Any:
    """Evaluate histogram operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Histogram", *args, **kwargs)


def histogram2d(*args: Any, **kwargs: Any) -> Any:
    """Evaluate histogram2d operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Histogram2d", *args, **kwargs)


def histogram_bin_edges(*args: Any, **kwargs: Any) -> Any:
    """Provide function to calculate only the edges of the bins used by the histogram function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HistogramBinEdges", *args, **kwargs)


def histogramdd(*args: Any, **kwargs: Any) -> Any:
    """Evaluate histogramdd operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Histogramdd", *args, **kwargs)


def i0(*args: Any, **kwargs: Any) -> Any:
    """Modify Bessel function of the first kind, order 0.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("I0", *args, **kwargs)


def interp(*args: Any, **kwargs: Any) -> Any:
    """One-dimensional linear interpolation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Interp", *args, **kwargs)


def median(*args: Any, **kwargs: Any) -> Any:
    """Evaluate median operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Median", *args, **kwargs)


def modf(*args: Any, **kwargs: Any) -> Any:
    """Return the fractional and integral parts of an array, element-wise.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Modf", *args, **kwargs)


def piecewise(*args: Any, **kwargs: Any) -> Any:
    """Evaluate piecewise operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Piecewise", *args, **kwargs)


def trapezoid(*args: Any, **kwargs: Any) -> Any:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Trapezoid", *args, **kwargs)


def indices(*args: Any, **kwargs: Any) -> Any:
    """Return an array representing the indices of a grid.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Indices", *args, **kwargs)


def ix_(*args: Any, **kwargs: Any) -> Any:
    """Construct an open mesh from multiple sequences.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Ix", *args, **kwargs)


def kron(*args: Any, **kwargs: Any) -> Any:
    """Kronecker product of two arrays.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Kron", *args, **kwargs)


def mask_indices(*args: Any, **kwargs: Any) -> Any:
    """Return the indices to access (n, n) arrays.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("MaskIndices", *args, **kwargs)


def rot90(*args: Any, **kwargs: Any) -> Any:
    """Rotate an array by 90 degrees in the plane specified by axes.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Rot90", *args, **kwargs)


def tri(*args: Any, **kwargs: Any) -> Any:
    """Provide an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Tri", *args, **kwargs)


def tril(*args: Any, **kwargs: Any) -> Any:
    """Lower triangle of an array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Tril", *args, **kwargs)


def trim_zeros(*args: Any, **kwargs: Any) -> Any:
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("TrimZeros", *args, **kwargs)


def triu(*args: Any, **kwargs: Any) -> Any:
    """Upper triangle of an array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Triu", *args, **kwargs)


def vander(*args: Any, **kwargs: Any) -> Any:
    """Generate a Vandermonde matrix.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Vander", *args, **kwargs)


def intersect1d(*args: Any, **kwargs: Any) -> Any:
    """Find the intersection of two arrays.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Intersect1d", *args, **kwargs)


def iinfo(*args: Any, **kwargs: Any) -> Any:
    """Iinfo operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iinfo", *args, **kwargs)


def isscalar(*args: Any, **kwargs: Any) -> Any:
    """Return True if the type of num is a scalar type.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isscalar", *args, **kwargs)


def iterable(*args: Any, **kwargs: Any) -> Any:
    """Check whether or not an object can be iterated over.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iterable", *args, **kwargs)


def promote_types(*args: Any, **kwargs: Any) -> Any:
    """Return the data type with the smallest size and smallest scalar kind.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PromoteTypes", *args, **kwargs)


def result_type(*args: Any, **kwargs: Any) -> Any:
    """Return the type that results from applying the NumPy type promotion rules.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("ResultType", *args, **kwargs)


def infeed(*args: Any, **kwargs: Any) -> Any:
    """Read from the infeed queue.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Infeed", *args, **kwargs)


def get_printoptions(*args: Any, **kwargs: Any) -> Any:
    """Get the current print options.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("GetPrintoptions", *args, **kwargs)


def unwrap(*args: Any, **kwargs: Any) -> Any:
    """Unwrap an array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Unwrap", *args, **kwargs)


def vectorize(*args: Any, **kwargs: Any) -> Any:
    """Vectorize a python function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Vectorize", *args, **kwargs)
