"""Misc operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Finfo")
class Finfo(OpDef):
    """Finfo operation."""

    op_name = "Finfo"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Iinfo")
class Iinfo(OpDef):
    """Iinfo operation."""

    op_name = "Iinfo"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("GetPrintoptions")
class GetPrintoptions(OpDef):
    """Get the current print options."""

    op_name = "GetPrintoptions"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Gradient")
class Gradient(OpDef):
    """Return the gradient of an N-dimensional array."""

    op_name = "Gradient"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Histogram")
class Histogram(OpDef):
    """Compute the histogram of a dataset."""

    op_name = "Histogram"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Histogram2d")
class Histogram2d(OpDef):
    """Compute the bi-dimensional histogram of two data samples."""

    op_name = "Histogram2d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("HistogramBinEdges")
class HistogramBinEdges(OpDef):
    """Function to calculate only the edges of the bins used by the histogram function."""

    op_name = "HistogramBinEdges"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Histogramdd")
class Histogramdd(OpDef):
    """Compute the multidimensional histogram of some data."""

    op_name = "Histogramdd"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("I0")
class I0(OpDef):
    """Modified Bessel function of the first kind, order 0."""

    op_name = "I0"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Indices")
class Indices(OpDef):
    """Return an array representing the indices of a grid."""

    op_name = "Indices"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Infeed")
class Infeed(OpDef):
    """Read from the infeed queue."""

    op_name = "Infeed"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Interp")
class Interp(OpDef):
    """One-dimensional linear interpolation."""

    op_name = "Interp"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Intersect1d")
class Intersect1d(OpDef):
    """Find the intersection of two arrays."""

    op_name = "Intersect1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Isscalar")
class Isscalar(OpDef):
    """Returns True if the type of num is a scalar type."""

    op_name = "Isscalar"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Iterable")
class Iterable(OpDef):
    """Check whether or not an object can be iterated over."""

    op_name = "Iterable"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Ix")
class Ix(OpDef):
    """Construct an open mesh from multiple sequences."""

    op_name = "Ix"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Kron")
class Kron(OpDef):
    """Kronecker product of two arrays."""

    op_name = "Kron"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("MaskIndices")
class MaskIndices(OpDef):
    """Return the indices to access (n, n) arrays."""

    op_name = "MaskIndices"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Median")
class Median(OpDef):
    """Compute the median along the specified axis."""

    op_name = "Median"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Mgrid")
class Mgrid(OpDef):
    """nd_grid instance which returns a dense multi-dimensional 'meshgrid'."""

    op_name = "Mgrid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Mish")
class Mish(OpDef):
    """Mish activation function."""

    op_name = "Mish"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Modf")
class Modf(OpDef):
    """Return the fractional and integral parts of an array, element-wise."""

    op_name = "Modf"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Ogrid")
class Ogrid(OpDef):
    """nd_grid instance which returns an open multi-dimensional 'meshgrid'."""

    op_name = "Ogrid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Piecewise")
class Piecewise(OpDef):
    """Evaluate a piecewise-defined function."""

    op_name = "Piecewise"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("PromoteTypes")
class PromoteTypes(OpDef):
    """Returns the data type with the smallest size and smallest scalar kind."""

    op_name = "PromoteTypes"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("R")
class R(OpDef):
    """Translates slice objects to concatenation along the first axis."""

    op_name = "R"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("ResultType")
class ResultType(OpDef):
    """Returns the type that results from applying the NumPy type promotion rules."""

    op_name = "ResultType"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Rot90")
class Rot90(OpDef):
    """Rotate an array by 90 degrees in the plane specified by axes."""

    op_name = "Rot90"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Trapezoid")
class Trapezoid(OpDef):
    """Integrate along the given axis using the composite trapezoidal rule."""

    op_name = "Trapezoid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Tri")
class Tri(OpDef):
    """An array with ones at and below the given diagonal and zeros elsewhere."""

    op_name = "Tri"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Tril")
class Tril(OpDef):
    """Lower triangle of an array."""

    op_name = "Tril"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("TrimZeros")
class TrimZeros(OpDef):
    """Trim the leading and/or trailing zeros from a 1-D array or sequence."""

    op_name = "TrimZeros"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Triu")
class Triu(OpDef):
    """Upper triangle of an array."""

    op_name = "Triu"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Unwrap")
class Unwrap(OpDef):
    """Unwrap by taking the complement of large deltas with respect to the period."""

    op_name = "Unwrap"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Vander")
class Vander(OpDef):
    """Generate a Vandermonde matrix."""

    op_name = "Vander"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Vectorize")
class Vectorize(OpDef):
    """Generalized function class."""

    op_name = "Vectorize"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("AxisIndex")
class AxisIndex(OpDef):
    """AxisIndex operation."""

    op_name = "AxisIndex"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()
