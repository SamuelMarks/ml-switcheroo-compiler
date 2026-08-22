# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module logical.py."""

from __future__ import annotations

"""Core abstractions and logic definitions for logical.py."""
from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .base import UnaryMathOp


@register_op("BitwiseNot")
class BitwiseNot(UnaryMathOp):
    """Compute bitwise NOT element-wise."""

    op_name = "BitwiseNot"
    np_op_name = "bitwise_not"


@register_op("Isfinite")
class Isfinite(UnaryMathOp):
    """Tests element-wise for finiteness (not infinity or NaN)."""

    op_name = "Isfinite"
    np_op_name = "isfinite"


@register_op("Isinf")
class Isinf(UnaryMathOp):
    """Tests element-wise for positive or negative infinity."""

    op_name = "Isinf"
    np_op_name = "isinf"


@register_op("Isnan")
class Isnan(UnaryMathOp):
    """Tests element-wise for NaN (Not a Number)."""

    op_name = "Isnan"
    np_op_name = "isnan"


@register_op("Isneginf")
class Isneginf(UnaryMathOp):
    """Tests element-wise for negative infinity."""

    op_name = "Isneginf"
    np_op_name = "isneginf"


@register_op("Isposinf")
class Isposinf(UnaryMathOp):
    """Tests element-wise for positive infinity."""

    op_name = "Isposinf"
    np_op_name = "isposinf"


@register_op("LogicalNot")
class LogicalNot(UnaryMathOp):
    """Compute the truth value of NOT x element-wise."""

    op_name = "LogicalNot"
    np_op_name = "logical_not"


@register_op("BitwiseCount")
class BitwiseCount(UnaryMathOp):
    """Compute the number of 1-bits in the binary representation of x."""

    op_name = "BitwiseCount"
    np_op_name = "bitwise_count"


@register_op("IsNonDecreasing")
class IsNonDecreasing(UnaryMathOp):
    """IsNonDecreasing operation."""

    op_name = "IsNonDecreasing"


@register_op("IsStrictlyIncreasing")
class IsStrictlyIncreasing(UnaryMathOp):
    """IsStrictlyIncreasing operation."""

    op_name = "IsStrictlyIncreasing"


@register_op("Packbits")
class Packbits(OpDef):
    """Packbits operator definition."""

    op_name = "Packbits"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("Unpackbits")
class Unpackbits(OpDef):
    """Unpackbits operator definition."""

    op_name = "Unpackbits"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("Clz")
class Clz(UnaryMathOp):
    """Count leading zeros."""

    op_name = "Clz"


@register_op("PopulationCount")
class PopulationCount(UnaryMathOp):
    """Population count."""

    op_name = "PopulationCount"


@register_op("BitcastConvertType")
class BitcastConvertType(UnaryMathOp):
    """Bitcast convert type."""

    op_name = "BitcastConvertType"


@register_op("ReducePrecision")
class ReducePrecision(UnaryMathOp):
    """Reduce precision."""

    op_name = "ReducePrecision"


@register_op("Iscomplex")
class Iscomplex(UnaryMathOp):
    """Tests element-wise for complex type."""

    op_name = "Iscomplex"
    np_op_name = "iscomplex"


@register_op("Isreal")
class Isreal(UnaryMathOp):
    """Tests element-wise for real type."""

    op_name = "Isreal"
    np_op_name = "isreal"


@register_op("Iscomplexobj")
class Iscomplexobj(OpDef):
    """Tests element-wise for complex object type."""

    op_name = "Iscomplexobj"
    np_op_name = "iscomplexobj"

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
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

    op_name = "Isrealobj"
    np_op_name = "isrealobj"

    def infer_shape(self, x: Any, **kwargs: Any) -> Any:
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

    op_name = "Issubdtype"
    np_op_name = "issubdtype"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
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

    op_name = "Isin"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        element = args[0] if len(args) > 0 else None
        return getattr(element, "shape", ())


def _get_size_from_shape(obj: Any) -> int | None:
    """Calculate the total number of elements from an object's shape.

    Args:
        obj (object): The object.

    Returns:
        int | None: The size.
    """
    shape = getattr(obj, "shape", ())
    if not shape:
        return 1
    size = 1
    for s in shape:
        if s is None:
            return None
        size *= s
    return size


@register_op("Ediff1d")
class Ediff1d(OpDef):
    """Differences between consecutive elements of an array."""

    op_name = "Ediff1d"
    np_op_name = "ediff1d"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:  # noqa: C901, PLR0912
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        ary = args[0] if len(args) > 0 else None
        to_end = kwargs.get("to_end")
        to_begin = kwargs.get("to_begin")

        size = _get_size_from_shape(ary)
        if size is not None:
            size -= 1

        if to_begin is not None:
            s_b = _get_size_from_shape(to_begin)
            size = size + s_b if size is not None and s_b is not None else None

        if to_end is not None:
            s_e = _get_size_from_shape(to_end)
            size = size + s_e if size is not None and s_e is not None else None

        return (size,)


def population_count(*args: Any, **kwargs: Any) -> Any:
    """Calculate element-wise population count (a.k.a. popcount, bitsum, bitcount).

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PopulationCount", *args, **kwargs)


def isin(*args: Any, **kwargs: Any) -> Any:
    """Calculate element in test_elements, broadcasting over element only.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isin", *args, **kwargs)


def iscomplex(*args: Any, **kwargs: Any) -> Any:
    """Return a bool array, where True if input element is complex.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iscomplex", *args, **kwargs)


def iscomplexobj(*args: Any, **kwargs: Any) -> Any:
    """Check for a complex type or an array of complex numbers.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iscomplexobj", *args, **kwargs)


def isreal(*args: Any, **kwargs: Any) -> Any:
    """Return a bool array, where True if input element is real.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isreal", *args, **kwargs)


def isrealobj(*args: Any, **kwargs: Any) -> Any:
    """Return True if x is a not complex type or an array of complex numbers.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isrealobj", *args, **kwargs)


def issubdtype(*args: Any, **kwargs: Any) -> Any:
    """Return True if first argument is a typecode lower/equal in type hierarchy.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Issubdtype", *args, **kwargs)


def reduce_precision(*args: Any, **kwargs: Any) -> Any:
    """Reduce precision operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("ReducePrecision", *args, **kwargs)
