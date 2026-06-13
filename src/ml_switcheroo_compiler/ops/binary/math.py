"""Defines binary mathematical operations and their shape inference and evaluation logic.

This module provides a base class `BinaryMathOp` and concrete implementations for
various element-wise binary mathematical operations (e.g., Add, Subtract, Multiply,
Divide) using NumPy for evaluation
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class BinaryMathOp(OpDef):
    """Base class for binary mathematical operations.

    This class defines the interface and common evaluation logic for operations
    that take two inputs and produce a single output
    """

    op_name: str = ""

    def infer_shape(self, x: object, y: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        # Broadcasting logic should ideally happen here, but for now we return x
        # This will be replaced by a proper shape inference pass
        return np.broadcast_shapes(x, y) if isinstance(x, tuple) and isinstance(y, tuple) else x

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return getattr(np, getattr(self, "np_op_name", self.op_name.lower()))(x, y)


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

    op_name = "True_Divide"

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.true_divide(x, y)


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
    """Computes x * log(y) returning 0 if x is 0 element-wise."""

    op_name = "Xlogy"

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import numpy as np

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        res = x_arr * np.log(y_arr)
        return np.where(x_arr == 0, np.zeros_like(res), res)
