# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module logical.py."""

from __future__ import annotations

"""Core abstractions and logic definitions for logical.py."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .base import UnaryMathOp


@register_op("BitwiseNot")
class BitwiseNot(UnaryMathOp):
    """Compute bitwise NOT element-wise."""

    op_name: object = "BitwiseNot"
    np_op_name: object = "bitwise_not"


@register_op("Isfinite")
class Isfinite(UnaryMathOp):
    """Tests element-wise for finiteness (not infinity or NaN)."""

    op_name: object = "Isfinite"
    np_op_name: object = "isfinite"


@register_op("Isinf")
class Isinf(UnaryMathOp):
    """Tests element-wise for positive or negative infinity."""

    op_name: object = "Isinf"
    np_op_name: object = "isinf"


@register_op("Isnan")
class Isnan(UnaryMathOp):
    """Tests element-wise for NaN (Not a Number)."""

    op_name: object = "Isnan"
    np_op_name: object = "isnan"


@register_op("Isneginf")
class Isneginf(UnaryMathOp):
    """Tests element-wise for negative infinity."""

    op_name: object = "Isneginf"
    np_op_name: object = "isneginf"


@register_op("Isposinf")
class Isposinf(UnaryMathOp):
    """Tests element-wise for positive infinity."""

    op_name: object = "Isposinf"
    np_op_name: object = "isposinf"


@register_op("LogicalNot")
class LogicalNot(UnaryMathOp):
    """Compute the truth value of NOT x element-wise."""

    op_name: object = "LogicalNot"
    np_op_name: object = "logical_not"


@register_op("BitwiseCount")
class BitwiseCount(UnaryMathOp):
    """Compute the number of 1-bits in the binary representation of x."""

    op_name: object = "BitwiseCount"
    np_op_name: object = "bitwise_count"


@register_op("IsNonDecreasing")
class IsNonDecreasing(UnaryMathOp):
    """IsNonDecreasing operation."""

    op_name: object = "IsNonDecreasing"


@register_op("IsStrictlyIncreasing")
class IsStrictlyIncreasing(UnaryMathOp):
    """IsStrictlyIncreasing operation."""

    op_name: object = "IsStrictlyIncreasing"


@register_op("Packbits")
class Packbits(OpDef):
    """Packbits operator definition."""

    op_name: object = "Packbits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


@register_op("Unpackbits")
class Unpackbits(OpDef):
    """Unpackbits operator definition."""

    op_name: object = "Unpackbits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


@register_op("Clz")
class Clz(UnaryMathOp):
    """Count leading zeros."""

    op_name: object = "Clz"


@register_op("PopulationCount")
class PopulationCount(UnaryMathOp):
    """Population count."""

    op_name: object = "PopulationCount"


@register_op("BitcastConvertType")
class BitcastConvertType(UnaryMathOp):
    """Bitcast convert type."""

    op_name: object = "BitcastConvertType"


@register_op("ReducePrecision")
class ReducePrecision(UnaryMathOp):
    """Reduce precision."""

    op_name: object = "ReducePrecision"


@register_op("Iscomplex")
class Iscomplex(UnaryMathOp):
    """Tests element-wise for complex type."""

    op_name: object = "Iscomplex"
    np_op_name: object = "iscomplex"


@register_op("Isreal")
class Isreal(UnaryMathOp):
    """Tests element-wise for real type."""

    op_name: object = "Isreal"
    np_op_name: object = "isreal"


@register_op("Iscomplexobj")
class Iscomplexobj(OpDef):
    """Tests element-wise for complex object type."""

    op_name: object = "Iscomplexobj"
    np_op_name: object = "iscomplexobj"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The x parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Isrealobj")
class Isrealobj(OpDef):
    """Tests element-wise for real object type."""

    op_name: object = "Isrealobj"
    np_op_name: object = "isrealobj"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The x parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Issubdtype")
class Issubdtype(OpDef):
    """Tests whether first argument is a typecode lower/equal in type hierarchy."""

    op_name: object = "Issubdtype"
    np_op_name: object = "issubdtype"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Isin")
class Isin(OpDef):
    """Calculate element in test_elements, broadcasting over element only."""

    op_name: object = "Isin"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        element: object = args[0] if len(args) > 0 else None
        return getattr(element, "shape", ())


def _get_size_from_shape(obj: object) -> int | None:
    """Calculate the total number of elements from an object's shape.

    Args:
        obj (object): The object.

    Returns:
        int | None: The size.
    """
    shape: object = getattr(obj, "shape", ())
    if not shape:
        return 1
    size: object = 1
    for s in shape:
        if s is None:
            return None
        size *= s
    return size


@register_op("Ediff1d")
class Ediff1d(OpDef):
    """Differences between consecutive elements of an array."""

    op_name: object = "Ediff1d"
    np_op_name: object = "ediff1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:  # noqa: C901, PLR0912
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        ary: object = args[0] if len(args) > 0 else None
        to_end: object = kwargs.get("to_end")
        to_begin: object = kwargs.get("to_begin")

        size: object = _get_size_from_shape(ary)
        if size is not None:
            size -= 1

        if to_begin is not None:
            s_b: object = _get_size_from_shape(to_begin)
            size: object = size + s_b if size is not None and s_b is not None else None

        if to_end is not None:
            s_e: object = _get_size_from_shape(to_end)
            size: object = size + s_e if size is not None and s_e is not None else None

        return (size,)


def population_count(*args: object, **kwargs: object) -> object:
    """Calculate element-wise population count (a.k.a. popcount, bitsum, bitcount).

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PopulationCount", *args, **kwargs)


def isin(*args: object, **kwargs: object) -> object:
    """Calculate element in test_elements, broadcasting over element only.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isin", *args, **kwargs)


def iscomplex(*args: object, **kwargs: object) -> object:
    """Return a bool array, where True if input element is complex.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iscomplex", *args, **kwargs)


def iscomplexobj(*args: object, **kwargs: object) -> object:
    """Check for a complex type or an array of complex numbers.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iscomplexobj", *args, **kwargs)


def isreal(*args: object, **kwargs: object) -> object:
    """Return a bool array, where True if input element is real.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isreal", *args, **kwargs)


def isrealobj(*args: object, **kwargs: object) -> object:
    """Return True if x is a not complex type or an array of complex numbers.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isrealobj", *args, **kwargs)


def issubdtype(*args: object, **kwargs: object) -> object:
    """Return True if first argument is a typecode lower/equal in type hierarchy.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Issubdtype", *args, **kwargs)


def reduce_precision(*args: object, **kwargs: object) -> object:
    """Reduce precision operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("ReducePrecision", *args, **kwargs)
