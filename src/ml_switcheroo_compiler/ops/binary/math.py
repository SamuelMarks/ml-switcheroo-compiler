"""Defines binary mathematical operations and their shape inference and evaluation logic.

This module provides a base class `BinaryMathOp` and concrete implementations for
various element-wise binary mathematical operations (e.g., Add, Subtract, Multiply,
Divide) using NumPy for evaluation
"""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class BinaryMathOp(OpDef):
    """Base class for binary mathematical operations.

    This class defines the interface and common evaluation logic for operations
    that take two inputs and produce a single output
    """

    op_name: str = ""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Execute infer_shape.

        Args:
            *shapes (Any): Argument *shapes.
            **kwargs (Any): Argument **kwargs.

        Returns:
        Any: The result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs

        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        # Broadcasting logic should ideally happen here, but for now we return x
        # This will be replaced by a proper shape inference pass
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()


@register_op("Add")
class Add(BinaryMathOp):
    """Binary operation for element-wise addition of two operands."""

    op_name = "Add"


@register_op("Subtract")
class Subtract(BinaryMathOp):
    """Binary operation for element-wise subtraction of the second operand from the first."""

    op_name = "Subtract"


@register_op("Multiply")
class Multiply(BinaryMathOp):
    """Binary operation for element-wise multiplication of two operands."""

    op_name = "Multiply"


@register_op("Divide")
class Divide(BinaryMathOp):
    """Binary operation for element-wise division of the first operand by the second."""

    op_name = "Divide"


@register_op("TrueDivide")
class TrueDivide(Divide):
    """Binary operation for element-wise true division of the first operand by the second."""

    op_name = "TrueDivide"


@register_op("Power")
class Power(BinaryMathOp):
    """Binary operation for element-wise exponentiation of the first operand to the power.

    of the second
    """

    op_name = "Power"


@register_op("Maximum")
class Maximum(BinaryMathOp):
    """Binary operation for element-wise maximum of two operands."""

    op_name = "Maximum"


@register_op("Minimum")
class Minimum(BinaryMathOp):
    """Binary operation for element-wise minimum of two operands."""

    op_name = "Minimum"


@register_op("BitwiseAnd")
class BitwiseAnd(BinaryMathOp):
    """Binary operation for element-wise bitwise AND of two integer operands."""

    op_name = "BitwiseAnd"
    np_op_name = "bitwise_and"


@register_op("BitwiseOr")
class BitwiseOr(BinaryMathOp):
    """Binary operation for element-wise bitwise OR of two integer operands."""

    op_name = "BitwiseOr"
    np_op_name = "bitwise_or"


@register_op("BitwiseXor")
class BitwiseXor(BinaryMathOp):
    """Binary operation for element-wise bitwise XOR of two integer operands."""

    op_name = "BitwiseXor"
    np_op_name = "bitwise_xor"


@register_op("Copysign")
class Copysign(BinaryMathOp):
    """Binary operation to copy the sign of the second operand to the first operand.

    element-wise
    """

    op_name = "Copysign"
    np_op_name = "copysign"


@register_op("FloatPower")
class FloatPower(BinaryMathOp):
    """Binary operation for element-wise exponentiation raised to non-integer powers."""

    op_name = "FloatPower"
    np_op_name = "float_power"


@register_op("FloorDivide")
class FloorDivide(BinaryMathOp):
    """Binary operation for element-wise floor division of the first operand by the second."""

    op_name = "FloorDivide"
    np_op_name = "floor_divide"


@register_op("Fmax")
class Fmax(BinaryMathOp):
    """Binary operation for element-wise maximum of two operands, ignoring NaNs."""

    op_name = "Fmax"
    np_op_name = "fmax"


@register_op("Fmin")
class Fmin(BinaryMathOp):
    """Binary operation for element-wise minimum of two operands, ignoring NaNs."""

    op_name = "Fmin"
    np_op_name = "fmin"


@register_op("Fmod")
class Fmod(BinaryMathOp):
    """Binary operation for element-wise remainder of division (fmod) matching the.

    platform

    C library
    """

    op_name = "Fmod"
    np_op_name = "fmod"


@register_op("Gcd")
class Gcd(BinaryMathOp):
    """Binary operation for element-wise greatest common divisor of two integer operands."""

    op_name = "Gcd"
    np_op_name = "gcd"


@register_op("Greater")
class Greater(BinaryMathOp):
    """Binary operation for element-wise greater-than comparison of two operands."""

    op_name = "Greater"
    np_op_name = "greater"


@register_op("GreaterEqual")
class GreaterEqual(BinaryMathOp):
    """Binary operation for element-wise greater-than-or-equal comparison of two operands."""

    op_name = "GreaterEqual"
    np_op_name = "greater_equal"


@register_op("Heaviside")
class Heaviside(BinaryMathOp):
    """Binary operation for element-wise Heaviside step function."""

    op_name = "Heaviside"
    np_op_name = "heaviside"


@register_op("Hypot")
class Hypot(BinaryMathOp):
    """Binary operation for element-wise hypotenuse calculation (sqrt(x1**2 + x2**2))."""

    op_name = "Hypot"
    np_op_name = "hypot"


@register_op("Lcm")
class Lcm(BinaryMathOp):
    """Binary operation for element-wise least common multiple of two integer operands."""

    op_name = "Lcm"
    np_op_name = "lcm"


@register_op("Ldexp")
class Ldexp(BinaryMathOp):
    """Binary operation for element-wise calculation of x * 2**y."""

    op_name = "Ldexp"
    np_op_name = "ldexp"


@register_op("LeftShift")
class LeftShift(BinaryMathOp):
    """Binary operation for element-wise left shift of the first operand by the second."""

    op_name = "LeftShift"
    np_op_name = "left_shift"


@register_op("Less")
class Less(BinaryMathOp):
    """Binary operation for element-wise less-than comparison of two operands."""

    op_name = "Less"
    np_op_name = "less"


@register_op("LessEqual")
class LessEqual(BinaryMathOp):
    """Binary operation for element-wise less-than-or-equal comparison of two operands."""

    op_name = "LessEqual"
    np_op_name = "less_equal"


@register_op("Logaddexp")
class Logaddexp(BinaryMathOp):
    """Binary operation for element-wise logarithm of the sum of exponentiations.

    (log(exp(x) + exp(y)))
    """

    op_name = "Logaddexp"
    np_op_name = "logaddexp"


@register_op("Logaddexp2")
class Logaddexp2(BinaryMathOp):
    """Binary operation for element-wise logarithm of the sum of exponentiations in base 2.

    (log2(2**x + 2**y))
    """

    op_name = "Logaddexp2"
    np_op_name = "logaddexp2"


@register_op("LogicalAnd")
class LogicalAnd(BinaryMathOp):
    """Binary operation for element-wise logical AND of two operands."""

    op_name = "LogicalAnd"
    np_op_name = "logical_and"


@register_op("LogicalOr")
class LogicalOr(BinaryMathOp):
    """Binary operation for element-wise logical OR of two operands."""

    op_name = "LogicalOr"
    np_op_name = "logical_or"


@register_op("LogicalXor")
class LogicalXor(BinaryMathOp):
    """Binary operation for element-wise logical XOR of two operands."""

    op_name = "LogicalXor"
    np_op_name = "logical_xor"


@register_op("Mod")
class Mod(BinaryMathOp):
    """Binary operation for element-wise modulo of two operands."""

    op_name = "Mod"
    np_op_name = "mod"


@register_op("Nextafter")
class Nextafter(BinaryMathOp):
    """Binary operation for element-wise next representable floating-point value after the.

    first operand toward the second
    """

    op_name = "Nextafter"
    np_op_name = "nextafter"


@register_op("NotEqual")
class NotEqual(BinaryMathOp):
    """Binary operation for element-wise not-equal comparison of two operands."""

    op_name = "NotEqual"
    np_op_name = "not_equal"


@register_op("Remainder")
class Remainder(BinaryMathOp):
    """Binary operation for element-wise remainder of division."""

    op_name = "Remainder"
    np_op_name = "remainder"


@register_op("RightShift")
class RightShift(BinaryMathOp):
    """Binary operation for element-wise right shift of the first operand by the second."""

    op_name = "RightShift"
    np_op_name = "right_shift"


@register_op("Equal")
class Equal(BinaryMathOp):
    """Binary operation for element-wise equality comparison of two operands."""

    op_name = "Equal"
    np_op_name = "equal"


@register_op("Xlogy")
class Xlogy(BinaryMathOp):
    """An operation class for computing x * log(y)."""

    op_name = "Xlogy"


@register_op("Igamma")
class Igamma(BinaryMathOp):
    """An operation class for computing the regularized lower incomplete gamma function."""

    op_name = "Igamma"


@register_op("Igammac")
class Igammac(BinaryMathOp):
    """An operation class for computing the regularized upper incomplete gamma function."""

    op_name = "Igammac"


@register_op("Zeta")
class Zeta(BinaryMathOp):
    """An operation class for computing the Hurwitz zeta function."""

    op_name = "Zeta"


@register_op("BesselJn")
class BesselJn(BinaryMathOp):
    """An operation class for computing the Bessel function of the first kind of real order and complex argument."""

    op_name = "BesselJn"


@register_op("Polygamma")
class Polygamma(BinaryMathOp):
    """An operation class for computing the polygamma function."""

    op_name = "Polygamma"


@register_op("Betainc")
class Betainc(OpDef):
    """Regularized incomplete beta function."""

    op_name = "Betainc"

    def infer_shape(self, a: object, b: object, x: object = None, **kwargs: object) -> object:
        """Infer shape."""
        # Simple broadcasting
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shape_a = getattr(a, "shape", ())  # pragma: no cover
        shape_b = getattr(b, "shape", ())  # pragma: no cover
        shape_x = getattr(x, "shape", ()) if x is not None else ()  # pragma: no cover
        return broadcast_shapes(broadcast_shapes(shape_a, shape_b), shape_x)  # pragma: no cover


@register_op("DivideNoNan")
class DivideNoNan(BinaryMathOp):
    """DivideNoNan operation."""

    op_name = "DivideNoNan"


@register_op("MultiplyNoNan")
class MultiplyNoNan(BinaryMathOp):
    """MultiplyNoNan operation."""

    op_name = "MultiplyNoNan"


@register_op("SquaredDifference")
class SquaredDifference(BinaryMathOp):
    """SquaredDifference operation."""

    op_name = "SquaredDifference"


@register_op("Xdivy")
class Xdivy(BinaryMathOp):
    """Xdivy operation."""

    op_name = "Xdivy"


@register_op("Xlog1py")
class Xlog1py(BinaryMathOp):
    """Xlog1py operation."""

    op_name = "Xlog1py"


@register_op("TruncateDiv")
class TruncateDiv(BinaryMathOp):
    """Binary operation for element-wise truncated division."""

    op_name = "TruncateDiv"


@register_op("TruncateMod")
class TruncateMod(BinaryMathOp):
    """Binary operation for element-wise truncated modulo."""

    op_name = "TruncateMod"


@register_op("Polyadd")
class Polyadd(OpDef):
    """Polyadd operator definition."""

    op_name = "Polyadd"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polysub")
class Polysub(OpDef):
    """Polysub operator definition."""

    op_name = "Polysub"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polymul")
class Polymul(OpDef):
    """Polymul operator definition."""

    op_name = "Polymul"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polydiv")
class Polydiv(OpDef):
    """Polydiv operator definition."""

    op_name = "Polydiv"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polyval")
class Polyval(OpDef):
    """Polyval operator definition."""

    op_name = "Polyval"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Poly")
class Poly(OpDef):
    """Poly operator definition."""

    op_name = "Poly"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polyder")
class Polyder(OpDef):
    """Polyder operator definition."""

    op_name = "Polyder"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polyfit")
class Polyfit(OpDef):
    """Polyfit operator definition."""

    op_name = "Polyfit"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Polyint")
class Polyint(OpDef):
    """Polyint operator definition."""

    op_name = "Polyint"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Roots")
class Roots(OpDef):
    """Roots operator definition."""

    op_name = "Roots"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("IgammaGradA")
class IgammaGradA(BinaryMathOp):
    """An operation class for computing the gradient of the regularized lower incomplete gamma function with respect to a."""

    op_name = "IgammaGradA"


@register_op("RandomGammaGrad")
class RandomGammaGrad(BinaryMathOp):
    """An operation class for computing the gradient of random_gamma with respect to alpha."""

    op_name = "RandomGammaGrad"


@register_op("SortKeyVal")
class SortKeyVal(BinaryMathOp):
    """Sort keys and values."""

    op_name = "SortKeyVal"


@register_op("Atan2")
class Atan2(BinaryMathOp):
    """Binary operation for element-wise arctangent of y/x."""

    op_name = "Atan2"
