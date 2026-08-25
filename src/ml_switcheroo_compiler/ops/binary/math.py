# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define binary mathematical operations and their shape inference and evaluation logic.

This module provides a base class `BinaryMathOp` and concrete implementations for
various element-wise binary mathematical operations (e.g., Add, Subtract, Multiply,
Divide) using NumPy for evaluation
"""

# Simple broadcasting
from ml_switcheroo_compiler.core.shape import broadcast_shapes
from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs
from ml_switcheroo_compiler.ops.base import OpDef, register_op


class BinaryMathOp(OpDef):
    """Define base class for binary mathematical operations.

    This class defines the interface and common evaluation logic for operations
    that take two inputs and produce a single output
    """

    op_name: str = ""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *shapes (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if shapes and all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes) if len(shapes) > 1 else shapes[0]
        return shapes[0] if shapes else ()


@register_op("Add")
class Add(BinaryMathOp):
    """Binary operation for element-wise addition of two operands."""

    op_name: object = "Add"


@register_op("Subtract")
class Subtract(BinaryMathOp):
    """Binary operation for element-wise subtraction of the second operand from the first."""

    op_name: object = "Subtract"


@register_op("Multiply")
class Multiply(BinaryMathOp):
    """Binary operation for element-wise multiplication of two operands."""

    op_name: object = "Multiply"


@register_op("Divide")
class Divide(BinaryMathOp):
    """Binary operation for element-wise division of the first operand by the second."""

    op_name: object = "Divide"


@register_op("TrueDivide")
class TrueDivide(Divide):
    """Binary operation for element-wise true division of the first operand by the second."""

    op_name: object = "TrueDivide"


@register_op("Power")
class Power(BinaryMathOp):
    """Binary operation for element-wise exponentiation of the first operand to the power.

    of the second
    """

    op_name: object = "Power"


@register_op("Maximum")
class Maximum(BinaryMathOp):
    """Binary operation for element-wise maximum of two operands."""

    op_name: object = "Maximum"


@register_op("Minimum")
class Minimum(BinaryMathOp):
    """Binary operation for element-wise minimum of two operands."""

    op_name: object = "Minimum"


@register_op("BitwiseAnd")
class BitwiseAnd(BinaryMathOp):
    """Binary operation for element-wise bitwise AND of two integer operands."""

    op_name: object = "BitwiseAnd"
    np_op_name: object = "bitwise_and"


@register_op("BitwiseOr")
class BitwiseOr(BinaryMathOp):
    """Binary operation for element-wise bitwise OR of two integer operands."""

    op_name: object = "BitwiseOr"
    np_op_name: object = "bitwise_or"


@register_op("BitwiseXor")
class BitwiseXor(BinaryMathOp):
    """Binary operation for element-wise bitwise XOR of two integer operands."""

    op_name: object = "BitwiseXor"
    np_op_name: object = "bitwise_xor"


@register_op("Copysign")
class Copysign(BinaryMathOp):
    """Binary operation to copy the sign of the second operand to the first operand.

    element-wise
    """

    op_name: object = "Copysign"
    np_op_name: object = "copysign"


@register_op("FloatPower")
class FloatPower(BinaryMathOp):
    """Binary operation for element-wise exponentiation raised to non-integer powers."""

    op_name: object = "FloatPower"
    np_op_name: object = "float_power"


@register_op("FloorDivide")
class FloorDivide(BinaryMathOp):
    """Binary operation for element-wise floor division of the first operand by the second."""

    op_name: object = "FloorDivide"
    np_op_name: object = "floor_divide"


@register_op("Fmax")
class Fmax(BinaryMathOp):
    """Binary operation for element-wise maximum of two operands, ignoring NaNs."""

    op_name: object = "Fmax"
    np_op_name: object = "fmax"


@register_op("Fmin")
class Fmin(BinaryMathOp):
    """Binary operation for element-wise minimum of two operands, ignoring NaNs."""

    op_name: object = "Fmin"
    np_op_name: object = "fmin"


@register_op("Fmod")
class Fmod(BinaryMathOp):
    """Binary operation for element-wise remainder of division (fmod) matching the.

    platform

    C library
    """

    op_name: object = "Fmod"
    np_op_name: object = "fmod"


@register_op("Gcd")
class Gcd(BinaryMathOp):
    """Binary operation for element-wise greatest common divisor of two integer operands."""

    op_name: object = "Gcd"
    np_op_name: object = "gcd"


@register_op("Greater")
class Greater(BinaryMathOp):
    """Binary operation for element-wise greater-than comparison of two operands."""

    op_name: object = "Greater"
    np_op_name: object = "greater"


@register_op("GreaterEqual")
class GreaterEqual(BinaryMathOp):
    """Binary operation for element-wise greater-than-or-equal comparison of two operands."""

    op_name: object = "GreaterEqual"
    np_op_name: object = "greater_equal"


@register_op("Heaviside")
class Heaviside(BinaryMathOp):
    """Binary operation for element-wise Heaviside step function."""

    op_name: object = "Heaviside"
    np_op_name: object = "heaviside"


@register_op("Hypot")
class Hypot(BinaryMathOp):
    """Binary operation for element-wise hypotenuse calculation (sqrt(x1**2 + x2**2))."""

    op_name: object = "Hypot"
    np_op_name: object = "hypot"


@register_op("Lcm")
class Lcm(BinaryMathOp):
    """Binary operation for element-wise least common multiple of two integer operands."""

    op_name: object = "Lcm"
    np_op_name: object = "lcm"


@register_op("Ldexp")
class Ldexp(BinaryMathOp):
    """Binary operation for element-wise calculation of x * 2**y."""

    op_name: object = "Ldexp"
    np_op_name: object = "ldexp"


@register_op("LeftShift")
class LeftShift(BinaryMathOp):
    """Binary operation for element-wise left shift of the first operand by the second."""

    op_name: object = "LeftShift"
    np_op_name: object = "left_shift"


@register_op("Less")
class Less(BinaryMathOp):
    """Binary operation for element-wise less-than comparison of two operands."""

    op_name: object = "Less"
    np_op_name: object = "less"


@register_op("LessEqual")
class LessEqual(BinaryMathOp):
    """Binary operation for element-wise less-than-or-equal comparison of two operands."""

    op_name: object = "LessEqual"
    np_op_name: object = "less_equal"


@register_op("Logaddexp")
class Logaddexp(BinaryMathOp):
    """Binary operation for element-wise logarithm of the sum of exponentiations.

    (log(exp(x) + exp(y)))
    """

    op_name: object = "Logaddexp"
    np_op_name: object = "logaddexp"


@register_op("Logaddexp2")
class Logaddexp2(BinaryMathOp):
    """Binary operation for element-wise logarithm of the sum of exponentiations in base 2.

    (log2(2**x + 2**y))
    """

    op_name: object = "Logaddexp2"
    np_op_name: object = "logaddexp2"


@register_op("LogicalAnd")
class LogicalAnd(BinaryMathOp):
    """Binary operation for element-wise logical AND of two operands."""

    op_name: object = "LogicalAnd"
    np_op_name: object = "logical_and"


@register_op("LogicalOr")
class LogicalOr(BinaryMathOp):
    """Binary operation for element-wise logical OR of two operands."""

    op_name: object = "LogicalOr"
    np_op_name: object = "logical_or"


@register_op("LogicalXor")
class LogicalXor(BinaryMathOp):
    """Binary operation for element-wise logical XOR of two operands."""

    op_name: object = "LogicalXor"
    np_op_name: object = "logical_xor"


@register_op("Mod")
class Mod(BinaryMathOp):
    """Binary operation for element-wise modulo of two operands."""

    op_name: object = "Mod"
    np_op_name: object = "mod"


@register_op("Nextafter")
class Nextafter(BinaryMathOp):
    """Binary operation for element-wise next representable floating-point value after the.

    first operand toward the second
    """

    op_name: object = "Nextafter"
    np_op_name: object = "nextafter"


@register_op("NotEqual")
class NotEqual(BinaryMathOp):
    """Binary operation for element-wise not-equal comparison of two operands."""

    op_name: object = "NotEqual"
    np_op_name: object = "not_equal"


@register_op("Remainder")
class Remainder(BinaryMathOp):
    """Binary operation for element-wise remainder of division."""

    op_name: object = "Remainder"
    np_op_name: object = "remainder"


@register_op("Rem")
class Rem(BinaryMathOp):
    """Binary operation for element-wise floating-point remainder."""

    op_name: object = "Rem"
    np_op_name: object = "fmod"


@register_op("RightShift")
class RightShift(BinaryMathOp):
    """Binary operation for element-wise right shift of the first operand by the second."""

    op_name: object = "RightShift"
    np_op_name: object = "right_shift"


@register_op("Equal")
class Equal(BinaryMathOp):
    """Binary operation for element-wise equality comparison of two operands."""

    op_name: object = "Equal"
    np_op_name: object = "equal"


@register_op("Diff")
class Diff(OpDef):
    """Diff operation."""

    op_name: object = "Diff"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Typically out_shape is similar to input, maybe -1 on one axis
        return ()


@register_op("Digitize")
class Digitize(OpDef):
    """Digitize operation."""

    op_name: object = "Digitize"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Returns indices matching input shape
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


@register_op("ArrayEquiv")
class ArrayEquiv(OpDef):
    """ArrayEquiv operation."""

    op_name: object = "ArrayEquiv"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Xlogy")
class Xlogy(BinaryMathOp):
    """Provide an operation class for computing x * log(y)."""

    op_name: object = "Xlogy"


@register_op("Igamma")
class Igamma(BinaryMathOp):
    """Provide an operation class for computing the regularized lower incomplete gamma function."""

    op_name: object = "Igamma"


@register_op("Igammac")
class Igammac(BinaryMathOp):
    """Provide an operation class for computing the regularized upper incomplete gamma function."""

    op_name: object = "Igammac"


@register_op("Zeta")
class Zeta(BinaryMathOp):
    """Provide an operation class for computing the Hurwitz zeta function."""

    op_name: object = "Zeta"


@register_op("BesselJn")
class BesselJn(BinaryMathOp):
    """Provide an operation class for computing the Bessel function of the first kind of real order and complex argument."""

    op_name: object = "BesselJn"


@register_op("Polygamma")
class Polygamma(BinaryMathOp):
    """Provide an operation class for computing the polygamma function."""

    op_name: object = "Polygamma"


@register_op("Betainc")
class Betainc(OpDef):
    """Regularized incomplete beta function."""

    op_name: object = "Betainc"

    def infer_shape(self, a: object, b: object, x: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a (object): The a parameter.
            b (object): The b parameter.
            x (object): The x parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape_a: object = getattr(a, "shape", ())
        shape_b: object = getattr(b, "shape", ())
        shape_x: object = getattr(x, "shape", ()) if x is not None else ()
        return broadcast_shapes(broadcast_shapes(shape_a, shape_b), shape_x)


@register_op("DivideNoNan")
class DivideNoNan(BinaryMathOp):
    """DivideNoNan operation."""

    op_name: object = "DivideNoNan"


@register_op("MultiplyNoNan")
class MultiplyNoNan(BinaryMathOp):
    """MultiplyNoNan operation."""

    op_name: object = "MultiplyNoNan"


@register_op("SquaredDifference")
class SquaredDifference(BinaryMathOp):
    """SquaredDifference operation."""

    op_name: object = "SquaredDifference"


@register_op("Xdivy")
class Xdivy(BinaryMathOp):
    """Xdivy operation."""

    op_name: object = "Xdivy"


@register_op("Xlog1py")
class Xlog1py(BinaryMathOp):
    """Xlog1py operation."""

    op_name: object = "Xlog1py"


@register_op("TruncateDiv")
class TruncateDiv(BinaryMathOp):
    """Binary operation for element-wise truncated division."""

    op_name: object = "TruncateDiv"


@register_op("TruncateMod")
class TruncateMod(BinaryMathOp):
    """Binary operation for element-wise truncated modulo."""

    op_name: object = "TruncateMod"


@register_op("Polyadd")
class Polyadd(BinaryMathOp):
    """Polyadd operator definition."""

    op_name: object = "Polyadd"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polysub")
class Polysub(BinaryMathOp):
    """Polysub operator definition."""

    op_name: object = "Polysub"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polymul")
class Polymul(BinaryMathOp):
    """Polymul operator definition."""

    op_name: object = "Polymul"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polydiv")
class Polydiv(BinaryMathOp):
    """Polydiv operator definition."""

    op_name: object = "Polydiv"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polyval")
class Polyval(BinaryMathOp):
    """Polyval operator definition."""

    op_name: object = "Polyval"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Poly")
class Poly(BinaryMathOp):
    """Poly operator definition."""

    op_name: object = "Poly"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polyder")
class Polyder(BinaryMathOp):
    """Polyder operator definition."""

    op_name: object = "Polyder"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polyfit")
class Polyfit(BinaryMathOp):
    """Polyfit operator definition."""

    op_name: object = "Polyfit"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Polyint")
class Polyint(BinaryMathOp):
    """Polyint operator definition."""

    op_name: object = "Polyint"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("Roots")
class Roots(BinaryMathOp):
    """Roots operator definition."""

    op_name: object = "Roots"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: object: Computed shape.
        """
        if args:
            if hasattr(args[0], "shape"):
                from collections import namedtuple

                ShapeHolder: object = namedtuple("ShapeHolder", ["shape"])
                return ShapeHolder(getattr(args[0], "shape", ()))
            return args[0]
        return ()


@register_op("IgammaGradA")
class IgammaGradA(BinaryMathOp):
    """Provide an operation class for computing the gradient of the regularized lower incomplete gamma function with respect to a."""

    op_name: object = "IgammaGradA"


@register_op("RandomGammaGrad")
class RandomGammaGrad(BinaryMathOp):
    """Provide an operation class for computing the gradient of random_gamma with respect to alpha."""

    op_name: object = "RandomGammaGrad"


@register_op("SortKeyVal")
class SortKeyVal(BinaryMathOp):
    """Sort keys and values."""

    op_name: object = "SortKeyVal"


@register_op("Atan2")
class Atan2(BinaryMathOp):
    """Binary operation for element-wise arctangent of y/x."""

    op_name: object = "Atan2"


@register_op("Clip")
class Clip(OpDef):
    """Operator Clip."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


def clip(x: object, min_val: object = None, max_val: object = None, **kwargs: object) -> object:
    """Clip values in a tensor.

    Args:
        x (object): The x parameter.
        min_val (object): The min_val parameter.
        max_val (object): The max_val parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Clip", x, min=min_val, max=max_val, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    kwargs["a_min"] = min_val
    kwargs["a_max"] = max_val
    return emit_ir_node(None, "Clip", [x], getattr(x, "shape_metadata", None), kwargs)


def rem(*args: object, **kwargs: object) -> object:
    """Evaluate rem operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("Rem")(*args, **kwargs)


@register_op("ChebyshevPolynomialT")
class ChebyshevPolynomialT(BinaryMathOp):
    """ChebyshevPolynomialT operation."""

    op_name: object = "ChebyshevPolynomialT"


@register_op("ChebyshevPolynomialU")
class ChebyshevPolynomialU(BinaryMathOp):
    """ChebyshevPolynomialU operation."""

    op_name: object = "ChebyshevPolynomialU"


@register_op("ShiftedChebyshevPolynomialT")
class ShiftedChebyshevPolynomialT(BinaryMathOp):
    """ShiftedChebyshevPolynomialT operation."""

    op_name: object = "ShiftedChebyshevPolynomialT"


@register_op("ShiftedChebyshevPolynomialU")
class ShiftedChebyshevPolynomialU(BinaryMathOp):
    """ShiftedChebyshevPolynomialU operation."""

    op_name: object = "ShiftedChebyshevPolynomialU"


@register_op("ShiftedChebyshevPolynomialV")
class ShiftedChebyshevPolynomialV(BinaryMathOp):
    """ShiftedChebyshevPolynomialV operation."""

    op_name: object = "ShiftedChebyshevPolynomialV"


@register_op("ShiftedChebyshevPolynomialW")
class ShiftedChebyshevPolynomialW(BinaryMathOp):
    """ShiftedChebyshevPolynomialW operation."""

    op_name: object = "ShiftedChebyshevPolynomialW"


@register_op("HermitePolynomialH")
class HermitePolynomialH(BinaryMathOp):
    """HermitePolynomialH operation."""

    op_name: object = "HermitePolynomialH"


@register_op("HermitePolynomialHe")
class HermitePolynomialHe(BinaryMathOp):
    """HermitePolynomialHe operation."""

    op_name: object = "HermitePolynomialHe"


@register_op("LaguerrePolynomialL")
class LaguerrePolynomialL(BinaryMathOp):
    """LaguerrePolynomialL operation."""

    op_name: object = "LaguerrePolynomialL"


@register_op("LegendrePolynomialP")
class LegendrePolynomialP(BinaryMathOp):
    """LegendrePolynomialP operation."""

    op_name: object = "LegendrePolynomialP"


def igamma_grad_a(*args: object, **kwargs: object) -> object:
    """Compute the gradient of the regularized incomplete gamma function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("IgammaGradA", *args, **kwargs)


def random_gamma_grad(*args: object, **kwargs: object) -> object:
    """Compute the derivative of a Gamma random variable.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("RandomGammaGrad", *args, **kwargs)


def sort_key_val(*args: object, **kwargs: object) -> object:
    """Sort keys and values.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("SortKeyVal", *args, **kwargs)
