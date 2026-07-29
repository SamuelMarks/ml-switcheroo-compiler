"""Core abstractions and logic definitions for logical.py."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .base import UnaryMathOp


@register_op("BitwiseNot")
class BitwiseNot(UnaryMathOp):
    """Computes bitwise NOT element-wise."""

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
    """Computes the truth value of NOT x element-wise."""

    op_name = "LogicalNot"
    np_op_name = "logical_not"


@register_op("BitwiseCount")
class BitwiseCount(UnaryMathOp):
    """Computes the number of 1-bits in the binary representation of x."""

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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
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

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Isrealobj")
class Isrealobj(OpDef):
    """Tests element-wise for real object type."""

    op_name = "Isrealobj"
    np_op_name = "isrealobj"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Issubdtype")
class Issubdtype(OpDef):
    """Tests whether first argument is a typecode lower/equal in type hierarchy."""

    op_name = "Issubdtype"
    np_op_name = "issubdtype"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Isin")
class Isin(OpDef):
    """Calculates element in test_elements, broadcasting over element only."""

    op_name = "Isin"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        element = args[0] if len(args) > 0 else None
        return getattr(element, "shape", ())


@register_op("Ediff1d")
class Ediff1d(OpDef):
    """Differences between consecutive elements of an array."""

    op_name = "Ediff1d"
    np_op_name = "ediff1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:  # noqa: C901, PLR0912
        """Infer shape."""
        ary = args[0] if len(args) > 0 else None
        to_end = kwargs.get("to_end")
        to_begin = kwargs.get("to_begin")
        shape = getattr(ary, "shape", ())
        size = 1
        for s in shape:
            if s is None:
                size = None
                break
            size *= s

        if size is not None:
            size -= 1

        if to_begin is not None:
            begin_shape = getattr(to_begin, "shape", ())
            if not begin_shape:
                size = size + 1 if size is not None else None
            else:
                s_b = 1
                for s in begin_shape:
                    if s is None:
                        s_b = None
                        break
                    s_b *= s
                size = size + s_b if size is not None and s_b is not None else None

        if to_end is not None:
            end_shape = getattr(to_end, "shape", ())
            if not end_shape:
                size = size + 1 if size is not None else None
            else:
                s_e = 1
                for s in end_shape:
                    if s is None:
                        s_e = None
                        break
                    s_e *= s
                size = size + s_e if size is not None and s_e is not None else None

        return (size,)


def population_count(*args: object, **kwargs: object) -> object:
    """Calculates element-wise population count (a.k.a. popcount, bitsum, bitcount)."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PopulationCount", *args, **kwargs)


def isin(*args: object, **kwargs: object) -> object:
    """Calculates element in test_elements, broadcasting over element only."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isin", *args, **kwargs)


def iscomplex(*args: object, **kwargs: object) -> object:
    """Returns a bool array, where True if input element is complex."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iscomplex", *args, **kwargs)


def iscomplexobj(*args: object, **kwargs: object) -> object:
    """Check for a complex type or an array of complex numbers."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iscomplexobj", *args, **kwargs)


def isreal(*args: object, **kwargs: object) -> object:
    """Returns a bool array, where True if input element is real."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isreal", *args, **kwargs)


def isrealobj(*args: object, **kwargs: object) -> object:
    """Return True if x is a not complex type or an array of complex numbers."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isrealobj", *args, **kwargs)


def issubdtype(*args: object, **kwargs: object) -> object:
    """Returns True if first argument is a typecode lower/equal in type hierarchy."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Issubdtype", *args, **kwargs)


def reduce_precision(*args: object, **kwargs: object) -> object:
    """Reduce precision operation."""
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("ReducePrecision", *args, **kwargs)
