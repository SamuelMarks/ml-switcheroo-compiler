"""Defines unary mathematical operations for the ML Switcheroo framework.

This module contains the base class `UnaryMathOp` and various concrete implementations
of unary mathematical operations (such as trigonometric, exponential, logarithmic, and
rounding functions) that can be evaluated using NumPy
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class UnaryMathOp(OpDef):
    """Base class for unary mathematical operations.

    Provides default implementations for shape inference and NumPy evaluation
    for operations that take a single input and apply an element-wise mathematical
    transformation

    Attributes:
    op_name (str): The name of the operation
    """

    op_name: str = ""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return x  # Unary ops typically preserve shape and dtype

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return getattr(np, getattr(self, "np_op_name", self.op_name.lower()))(x)


@register_op("Sin")
class Sin(UnaryMathOp):
    """Computes the trigonometric sine element-wise."""

    op_name = "Sin"


@register_op("Cos")
class Cos(UnaryMathOp):
    """Computes the trigonometric cosine element-wise."""

    op_name = "Cos"


@register_op("Exp")
class Exp(UnaryMathOp):
    """Computes the exponential of all elements in the input."""

    op_name = "Exp"


@register_op("Log")
class Log(UnaryMathOp):
    """Computes the natural logarithm element-wise."""

    op_name = "Log"


@register_op("Sqrt")
class Sqrt(UnaryMathOp):
    """Computes the non-negative square root element-wise."""

    op_name = "Sqrt"


@register_op("Square")
class Square(UnaryMathOp):
    """Computes the element-wise square of the input."""

    op_name = "Square"


@register_op("Abs")
class Abs(UnaryMathOp):
    """Computes the absolute value element-wise."""

    op_name = "Abs"


@register_op("Negative")
class Negative(UnaryMathOp):
    """Computes the numerical negative element-wise."""

    op_name = "Negative"


@register_op("Positive")
class Positive(UnaryMathOp):
    """Computes the numerical positive element-wise (identity operation)."""

    op_name = "Positive"


@register_op("Sign")
class Sign(UnaryMathOp):
    """Computes an element-wise indication of the sign of a number (-1, 0, or 1)."""

    op_name = "Sign"


@register_op("Floor")
class Floor(UnaryMathOp):
    """Computes the floor of the input element-wise."""

    op_name = "Floor"


@register_op("Ceil")
class Ceil(UnaryMathOp):
    """Computes the ceiling of the input element-wise."""

    op_name = "Ceil"


@register_op("Round")
class Round(UnaryMathOp):
    """Rounds values of the input to the nearest integer element-wise."""

    op_name = "Round"


@register_op("Acos")
class Acos(UnaryMathOp):
    """Computes the inverse cosine (arccosine) element-wise."""

    op_name = "Acos"
    np_op_name = "arccos"


@register_op("Acosh")
class Acosh(UnaryMathOp):
    """Computes the inverse hyperbolic cosine element-wise."""

    op_name = "Acosh"
    np_op_name = "arccosh"


@register_op("Asin")
class Asin(UnaryMathOp):
    """Computes the inverse sine (arcsine) element-wise."""

    op_name = "Asin"
    np_op_name = "arcsin"


@register_op("Asinh")
class Asinh(UnaryMathOp):
    """Computes the inverse hyperbolic sine element-wise."""

    op_name = "Asinh"
    np_op_name = "arcsinh"


@register_op("Atan")
class Atan(UnaryMathOp):
    """Computes the inverse tangent (arctangent) element-wise."""

    op_name = "Atan"
    np_op_name = "arctan"


@register_op("Atanh")
class Atanh(UnaryMathOp):
    """Computes the inverse hyperbolic tangent element-wise."""

    op_name = "Atanh"
    np_op_name = "arctanh"


@register_op("BitwiseNot")
class BitwiseNot(UnaryMathOp):
    """Computes bitwise NOT element-wise."""

    op_name = "BitwiseNot"
    np_op_name = "bitwise_not"


@register_op("Cbrt")
class Cbrt(UnaryMathOp):
    """Computes the cube root element-wise."""

    op_name = "Cbrt"
    np_op_name = "cbrt"


@register_op("Conj")
class Conj(UnaryMathOp):
    """Computes the complex conjugate element-wise."""

    op_name = "Conj"
    np_op_name = "conj"


@register_op("Cosh")
class Cosh(UnaryMathOp):
    """Computes the hyperbolic cosine element-wise."""

    op_name = "Cosh"
    np_op_name = "cosh"


@register_op("Deg2Rad")
class Deg2Rad(UnaryMathOp):
    """Converts angles from degrees to radians element-wise."""

    op_name = "Deg2Rad"
    np_op_name = "deg2rad"


@register_op("Erf")
class Erf(UnaryMathOp):
    """Computes the error function element-wise."""

    op_name = "Erf"
    np_op_name = "erf"

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import math

        import numpy as np

        return np.vectorize(math.erf)(x).astype(getattr(x, "dtype", float))


@register_op("Erfc")
class Erfc(UnaryMathOp):
    """Computes the complementary error function element-wise."""

    op_name = "Erfc"
    np_op_name = "erfc"

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import math

        import numpy as np

        return np.vectorize(math.erfc)(x).astype(getattr(x, "dtype", float))


@register_op("Erfinv")
class Erfinv(UnaryMathOp):
    """Computes the inverse error function element-wise."""

    op_name = "Erfinv"
    np_op_name = "erfinv"

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        msg = "No direct NumPy equivalent for erfinv."
        raise UnimplementedMathError(msg)


@register_op("Exp2")
class Exp2(UnaryMathOp):
    """Computes 2**x element-wise."""

    op_name = "Exp2"
    np_op_name = "exp2"


@register_op("Expm1")
class Expm1(UnaryMathOp):
    """Computes exp(x) - 1 element-wise."""

    op_name = "Expm1"
    np_op_name = "expm1"


@register_op("Fix")
class Fix(UnaryMathOp):
    """Rounds elements to the nearest integer towards zero."""

    op_name = "Fix"
    np_op_name = "fix"


@register_op("Imag")
class Imag(UnaryMathOp):
    """Returns the imaginary part of the complex argument element-wise."""

    op_name = "Imag"
    np_op_name = "imag"


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


@register_op("Lgamma")
class Lgamma(UnaryMathOp):
    """Computes the natural logarithm of the absolute value of the gamma function element-.

    wise
    """

    op_name = "Lgamma"
    np_op_name = "lgamma"

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import math

        import numpy as np

        return np.vectorize(math.lgamma)(x).astype(getattr(x, "dtype", float))


@register_op("Log10")
class Log10(UnaryMathOp):
    """Computes the base-10 logarithm element-wise."""

    op_name = "Log10"
    np_op_name = "log10"


@register_op("Log1P")
class Log1P(UnaryMathOp):
    """Computes natural logarithm of 1 + x element-wise."""

    op_name = "Log1P"
    np_op_name = "log1p"


@register_op("Log2")
class Log2(UnaryMathOp):
    """Computes the base-2 logarithm element-wise."""

    op_name = "Log2"
    np_op_name = "log2"


@register_op("LogicalNot")
class LogicalNot(UnaryMathOp):
    """Computes the truth value of NOT x element-wise."""

    op_name = "LogicalNot"
    np_op_name = "logical_not"


@register_op("Rad2Deg")
class Rad2Deg(UnaryMathOp):
    """Converts angles from radians to degrees element-wise."""

    op_name = "Rad2Deg"
    np_op_name = "rad2deg"


@register_op("Real")
class Real(UnaryMathOp):
    """Returns the real part of the complex argument element-wise."""

    op_name = "Real"
    np_op_name = "real"


@register_op("Reciprocal")
class Reciprocal(UnaryMathOp):
    """Computes the reciprocal (1/x) element-wise."""

    op_name = "Reciprocal"
    np_op_name = "reciprocal"


@register_op("Sinc")
class Sinc(UnaryMathOp):
    """Computes the normalized sinc function element-wise."""

    op_name = "Sinc"
    np_op_name = "sinc"


@register_op("Sinh")
class Sinh(UnaryMathOp):
    """Computes the hyperbolic sine element-wise."""

    op_name = "Sinh"
    np_op_name = "sinh"


@register_op("Tan")
class Tan(UnaryMathOp):
    """Computes the trigonometric tangent element-wise."""

    op_name = "Tan"
    np_op_name = "tan"


@register_op("Tanh")
class Tanh(UnaryMathOp):
    """Computes the hyperbolic tangent element-wise."""

    op_name = "Tanh"
    np_op_name = "tanh"


@register_op("Trunc")
class Trunc(UnaryMathOp):
    """Truncates the fractional part of the input element-wise."""

    op_name = "Trunc"
    np_op_name = "trunc"


@register_op("Digamma")
class Digamma(UnaryMathOp):
    """Computes the digamma function element-wise."""

    op_name = "Digamma"
    np_op_name = "digamma"

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        msg = "No direct NumPy equivalent for digamma."
        raise UnimplementedMathError(msg)


@register_op("Rsqrt")
class Rsqrt(UnaryMathOp):
    """Computes the reciprocal square root (1 / sqrt(x)) element-wise."""

    op_name = "Rsqrt"
    np_op_name = "rsqrt"

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        import numpy as np

        return 1.0 / np.sqrt(x)
