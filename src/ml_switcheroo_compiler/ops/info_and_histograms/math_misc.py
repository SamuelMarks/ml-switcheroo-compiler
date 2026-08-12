# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Misc operations."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


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


def gradient(*args: Any, **kwargs: Any) -> Any:
    """Return the gradient of an N-dimensional array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Gradient", *args, **kwargs)


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


def kron(*args: Any, **kwargs: Any) -> Any:
    """Kronecker product of two arrays.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Kron", *args, **kwargs)


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


def unwrap(*args: Any, **kwargs: Any) -> Any:
    """Unwrap an array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Unwrap", *args, **kwargs)
