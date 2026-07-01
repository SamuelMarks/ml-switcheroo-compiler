"""Defines unary mathematical operations for the ML Switcheroo framework.

This module contains the base class `UnaryMathOp` and various concrete implementations
of unary mathematical operations (such as trigonometric, exponential, logarithmic, and
rounding functions) that can be evaluated using NumPy
"""

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

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return shapes[0] if shapes else ()  # Unary ops typically preserve shape and dtype


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


@register_op("BesselI0e")
class BesselI0e(UnaryMathOp):
    """An operation class for computing the exponentially scaled modified Bessel function of order 0."""

    op_name = "BesselI0e"


@register_op("BesselI1e")
class BesselI1e(UnaryMathOp):
    """An operation class for computing the exponentially scaled modified Bessel function of order 1."""

    op_name = "BesselI1e"


@register_op("Erfc")
class Erfc(UnaryMathOp):
    """Computes the complementary error function element-wise."""

    op_name = "Erfc"
    np_op_name = "erfc"


@register_op("Erfinv")
class Erfinv(UnaryMathOp):
    """Computes the inverse error function element-wise."""

    op_name = "Erfinv"
    np_op_name = "erfinv"


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


@register_op("Rsqrt")
class Rsqrt(UnaryMathOp):
    """Computes the reciprocal square root (1 / sqrt(x)) element-wise."""

    op_name = "Rsqrt"
    np_op_name = "rsqrt"


@register_op("Logit")
class Logit(UnaryMathOp):
    """Computes the logit of a tensor element-wise."""

    op_name = "Logit"


@register_op("Mvlgamma")
class Mvlgamma(UnaryMathOp):
    """Computes the multivariate log-gamma function with dimension p element-wise."""

    op_name = "Mvlgamma"


@register_op("SpecialGamma")
class SpecialGamma(UnaryMathOp):
    """Computes the gamma function element-wise."""

    op_name = "SpecialGamma"
    np_op_name = "gamma"


@register_op("NanToNum")
class NanToNum(UnaryMathOp):
    """Replaces NaN, positive infinity, and negative infinity values."""

    op_name = "NanToNum"

    def __call__(self, x: object, **kwargs: object) -> object:
        """Call NanToNum, filtering out the copy kwarg."""
        kwargs.pop("copy", None)
        return super().__call__(x, **kwargs)


@register_op("Signbit")
class Signbit(UnaryMathOp):
    """Returns True where signbit is set (less than zero)."""

    op_name = "Signbit"
    np_op_name = "signbit"


@register_op("Angle")
class Angle(UnaryMathOp):
    """Computes the angle of the complex argument."""

    op_name = "Angle"


@register_op("BitwiseCount")
class BitwiseCount(UnaryMathOp):
    """Computes the number of 1-bits in the binary representation of x."""

    op_name = "BitwiseCount"
    np_op_name = "bitwise_count"


@register_op("BesselI0")
class BesselI0(UnaryMathOp):
    """Modified Bessel function of order 0."""

    op_name = "BesselI0"
    np_op_name = "i0"


@register_op("BesselI1")
class BesselI1(UnaryMathOp):
    """Modified Bessel function of order 1."""

    op_name = "BesselI1"


@register_op("Erfcinv")
class Erfcinv(UnaryMathOp):
    """Inverse complementary error function."""

    op_name = "Erfcinv"


@register_op("Ndtri")
class Ndtri(UnaryMathOp):
    """Inverse standard normal CDF."""

    op_name = "Ndtri"


@register_op("Lbeta")
class Lbeta(OpDef):
    """Log of the absolute value of the beta function."""

    op_name = "Lbeta"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape."""
        shape = getattr(x, "shape", ())
        if len(shape) > 0:  # pragma: no branch
            return shape[:-1]  # pragma: no cover
        return ()


@register_op("ReciprocalNoNan")
class ReciprocalNoNan(UnaryMathOp):
    """ReciprocalNoNan operation."""

    op_name = "ReciprocalNoNan"


@register_op("IsNonDecreasing")
class IsNonDecreasing(UnaryMathOp):
    """IsNonDecreasing operation."""

    op_name = "IsNonDecreasing"


@register_op("IsStrictlyIncreasing")
class IsStrictlyIncreasing(UnaryMathOp):
    """IsStrictlyIncreasing operation."""

    op_name = "IsStrictlyIncreasing"


@register_op("L2Normalize")
class L2Normalize(UnaryMathOp):
    """L2Normalize operation."""

    op_name = "L2Normalize"


@register_op("ZeroFraction")
class ZeroFraction(UnaryMathOp):
    """ZeroFraction operation."""

    op_name = "ZeroFraction"


@register_op("BesselJ0")
class BesselJ0(UnaryMathOp):
    """BesselJ0 operation."""

    op_name = "BesselJ0"


@register_op("BesselJ1")
class BesselJ1(UnaryMathOp):
    """BesselJ1 operation."""

    op_name = "BesselJ1"


@register_op("BesselK0")
class BesselK0(UnaryMathOp):
    """BesselK0 operation."""

    op_name = "BesselK0"


@register_op("BesselK0e")
class BesselK0e(UnaryMathOp):
    """BesselK0e operation."""

    op_name = "BesselK0e"


@register_op("BesselK1")
class BesselK1(UnaryMathOp):
    """BesselK1 operation."""

    op_name = "BesselK1"


@register_op("BesselK1e")
class BesselK1e(UnaryMathOp):
    """BesselK1e operation."""

    op_name = "BesselK1e"


@register_op("BesselY0")
class BesselY0(UnaryMathOp):
    """BesselY0 operation."""

    op_name = "BesselY0"


@register_op("BesselY1")
class BesselY1(UnaryMathOp):
    """BesselY1 operation."""

    op_name = "BesselY1"


@register_op("Dawsn")
class Dawsn(UnaryMathOp):
    """Dawsn operation."""

    op_name = "Dawsn"


@register_op("Expint")
class Expint(UnaryMathOp):
    """Expint operation."""

    op_name = "Expint"


@register_op("FresnelCos")
class FresnelCos(UnaryMathOp):
    """FresnelCos operation."""

    op_name = "FresnelCos"


@register_op("FresnelSin")
class FresnelSin(UnaryMathOp):
    """FresnelSin operation."""

    op_name = "FresnelSin"


@register_op("Spence")
class Spence(UnaryMathOp):
    """Spence operation."""

    op_name = "Spence"


@register_op("Setdiff1d")
class Setdiff1d(OpDef):
    """Setdiff1d operator definition."""

    op_name = "Setdiff1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Setxor1d")
class Setxor1d(OpDef):
    """Setxor1d operator definition."""

    op_name = "Setxor1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Union1d")
class Union1d(OpDef):
    """Union1d operator definition."""

    op_name = "Union1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueAll")
class UniqueAll(OpDef):
    """UniqueAll operator definition."""

    op_name = "UniqueAll"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueCounts")
class UniqueCounts(OpDef):
    """UniqueCounts operator definition."""

    op_name = "UniqueCounts"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueInverse")
class UniqueInverse(OpDef):
    """UniqueInverse operator definition."""

    op_name = "UniqueInverse"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueValues")
class UniqueValues(OpDef):
    """UniqueValues operator definition."""

    op_name = "UniqueValues"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Packbits")
class Packbits(OpDef):
    """Packbits operator definition."""

    op_name = "Packbits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Unpackbits")
class Unpackbits(OpDef):
    """Unpackbits operator definition."""

    op_name = "Unpackbits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


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
