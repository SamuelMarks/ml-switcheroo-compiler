"""Misc operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import coerce_int, coerce_operand_list
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Gradient")
class Gradient(OpDef):
    """Return the gradient of an N-dimensional array."""

    op_name = "Gradient"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("I0")
class I0(OpDef):
    """Modify Bessel function of the first kind, order 0."""

    op_name = "I0"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Interp")
class Interp(OpDef):
    """One-dimensional linear interpolation."""

    op_name = "Interp"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Intersect1d")
class Intersect1d(OpDef):
    """Find the intersection of two arrays."""

    op_name = "Intersect1d"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


@register_op("Kron")
class Kron(OpDef):
    """Kronecker product of two arrays."""

    op_name = "Kron"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if len(args) < 2:
            return ()
        a, b = args[0], args[1]
        shape_a = extract_shape(a)
        shape_b = extract_shape(b)
        ndims = max(len(shape_a), len(shape_b))
        shape_a = (1,) * (ndims - len(shape_a)) + shape_a
        shape_b = (1,) * (ndims - len(shape_b)) + shape_b
        return tuple(a_dim * b_dim for a_dim, b_dim in zip(shape_a, shape_b))


@register_op("Median")
class Median(OpDef):
    """Compute the median along the specified axis."""

    op_name = "Median"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        shape = list(extract_shape(args[0]))
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        if axis is None:
            return (1,) if keepdims else ()
        axes_list = [axis] if isinstance(axis, int) else [int(coerce_int(a)) for a in coerce_operand_list(axis)]
        for ax in sorted(axes_list, reverse=True):
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

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Modf")
class Modf(OpDef):
    """Return the fractional and integral parts of an array, element-wise."""

    op_name = "Modf"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Piecewise")
class Piecewise(OpDef):
    """Evaluate a piecewise-defined function."""

    op_name = "Piecewise"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Rot90")
class Rot90(OpDef):
    """Rotate an array by 90 degrees in the plane specified by axes."""

    op_name = "Rot90"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        shape = list(extract_shape(args[0]))
        raw_axes = kwargs.get("axes", (0, 1))
        axes_seq = raw_axes if isinstance(raw_axes, (list, tuple)) else coerce_operand_list(raw_axes)
        if (
            len(axes_seq) == 2
            and int(coerce_int(axes_seq[0])) < len(shape)
            and int(coerce_int(axes_seq[1])) < len(shape)
        ):
            a0, a1 = int(coerce_int(axes_seq[0])), int(coerce_int(axes_seq[1]))
            shape[a0], shape[a1] = shape[a1], shape[a0]
        return tuple(shape)


@register_op("Trapezoid")
class Trapezoid(OpDef):
    """Integrate along the given axis using the composite trapezoidal rule."""

    op_name = "Trapezoid"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        shape: list[int | str | None] = (
            [d for d in args[0] if isinstance(d, (int, str)) or d is None]
            if isinstance(args[0], tuple)
            else list(extract_shape(args[0]))
        )
        axis = int(coerce_int(kwargs.get("axis", -1)))
        if -len(shape) <= axis < len(shape):
            shape.pop(axis)
        return tuple(shape)


@register_op("Tri")
class Tri(OpDef):
    """Provide an array with ones at and below the given diagonal and zeros elsewhere."""

    op_name = "Tri"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        N = args[0] if args else 0
        M = kwargs.get("M", N)
        return (N, M)


@register_op("Tril")
class Tril(OpDef):
    """Lower triangle of an array."""

    op_name = "Tril"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("TrimZeros")
class TrimZeros(OpDef):
    """Trim the leading and/or trailing zeros from a 1-D array or sequence."""

    op_name = "TrimZeros"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return (None,)


@register_op("Triu")
class Triu(OpDef):
    """Upper triangle of an array."""

    op_name = "Triu"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Unwrap")
class Unwrap(OpDef):
    """Unwrap by taking the complement of large deltas with respect to the period."""

    op_name = "Unwrap"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        return extract_shape(args[0])


@register_op("Vander")
class Vander(OpDef):
    """Generate a Vandermonde matrix."""

    op_name = "Vander"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        x = extract_shape(args[0])
        N = kwargs.get("N", x[0] if x else 0)
        return (*x, N)


def gradient(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return the gradient of an N-dimensional array.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Gradient", *args, **kwargs)


def i0(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Modify Bessel function of the first kind, order 0.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("I0", *args, **kwargs)


def interp(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """One-dimensional linear interpolation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Interp", *args, **kwargs)


def median(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the median tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Median", *args, **kwargs)


def modf(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return the fractional and integral parts of an array, element-wise.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Modf", *args, **kwargs)


def piecewise(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the piecewise tensor operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Piecewise", *args, **kwargs)


def trapezoid(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Trapezoid", *args, **kwargs)


def kron(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Kronecker product of two arrays.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Kron", *args, **kwargs)


def rot90(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Rotate an array by 90 degrees in the plane specified by axes.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Rot90", *args, **kwargs)


def tri(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Provide an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Tri", *args, **kwargs)


def tril(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Lower triangle of an array.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Tril", *args, **kwargs)


def trim_zeros(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("TrimZeros", *args, **kwargs)


def triu(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Upper triangle of an array.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Triu", *args, **kwargs)


def vander(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Generate a Vandermonde matrix.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Vander", *args, **kwargs)


def intersect1d(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Find the intersection of two arrays.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Intersect1d", *args, **kwargs)


def unwrap(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Unwrap an array.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Unwrap", *args, **kwargs)
