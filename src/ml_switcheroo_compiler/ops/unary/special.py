"""Defines special unary operations for the ML Switcheroo framework, including Cast,.

Bitcast, and Frexp
"""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .base import UnaryMathOp


@register_op("Cast")
class Cast(OpDef):
    """An operation that casts an input array to a specified data type."""

    def infer_shape(self, x: object, dtype: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            dtype (object, optional): The target data type.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x


@register_op("Bitcast")
class Bitcast(Cast):
    """An operation that bitcasts an input array to a specified data type without copying."""


@register_op("Frexp")
class Frexp(OpDef):
    """An operation that decomposes a floating-point array into mantissa and exponent."""

    def infer_shape(self, x: object, dtype: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            dtype (object, optional): The target data type.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x


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


@register_op("Lgamma")
class Lgamma(UnaryMathOp):
    """Computes the natural logarithm of the absolute value of the gamma function element-.

    wise
    """

    op_name = "Lgamma"
    np_op_name = "lgamma"


@register_op("Digamma")
class Digamma(UnaryMathOp):
    """Computes the digamma function element-wise."""

    op_name = "Digamma"
    np_op_name = "digamma"


@register_op("Mvlgamma")
class Mvlgamma(UnaryMathOp):
    """Computes the multivariate log-gamma function with dimension p element-wise."""

    op_name = "Mvlgamma"


@register_op("SpecialGamma")
class SpecialGamma(UnaryMathOp):
    """Computes the gamma function element-wise."""

    op_name = "SpecialGamma"
    np_op_name = "gamma"


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
        if len(shape) > 0:
            return shape[:-1]
        return ()


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
