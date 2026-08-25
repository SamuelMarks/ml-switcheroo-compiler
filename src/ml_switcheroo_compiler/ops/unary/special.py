# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define special unary operations for the ML Switcheroo framework, including Cast,.

Bitcast, and Frexp
"""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .base import UnaryMathOp


@register_op("Cast")
class Cast(OpDef):
    """Provide an operation that casts an input array to a specified data type."""

    def infer_shape(self, x: object, dtype: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The x parameter.
            dtype (object): The dtype parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return x


@register_op("Bitcast")
class Bitcast(Cast):
    """Provide an operation that bitcasts an input array to a specified data type without copying."""


@register_op("CanCast")
class CanCast(OpDef):
    """CanCast operation."""

    op_name: object = "CanCast"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # CanCast returns a boolean scalar
        return ()


@register_op("Frexp")
class Frexp(OpDef):
    """Provide an operation that decomposes a floating-point array into mantissa and exponent."""

    def infer_shape(self, x: object, dtype: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter.
            dtype (object): The dtype parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return x


@register_op("Erf")
class Erf(UnaryMathOp):
    """Compute the error function element-wise."""

    op_name: object = "Erf"
    np_op_name: object = "erf"


@register_op("BesselI0e")
class BesselI0e(UnaryMathOp):
    """Provide an operation class for computing the exponentially scaled modified Bessel function of order 0."""

    op_name: object = "BesselI0e"


@register_op("BesselI1e")
class BesselI1e(UnaryMathOp):
    """Provide an operation class for computing the exponentially scaled modified Bessel function of order 1."""

    op_name: object = "BesselI1e"


@register_op("Erfc")
class Erfc(UnaryMathOp):
    """Compute the complementary error function element-wise."""

    op_name: object = "Erfc"
    np_op_name: object = "erfc"


@register_op("Erfinv")
class Erfinv(UnaryMathOp):
    """Compute the inverse error function element-wise."""

    op_name: object = "Erfinv"
    np_op_name: object = "erfinv"


@register_op("Lgamma")
class Lgamma(UnaryMathOp):
    """Compute the natural logarithm of the absolute value of the gamma function element-.

    wise
    """

    op_name: object = "Lgamma"
    np_op_name: object = "lgamma"


@register_op("Digamma")
class Digamma(UnaryMathOp):
    """Compute the digamma function element-wise."""

    op_name: object = "Digamma"
    np_op_name: object = "digamma"


@register_op("Mvlgamma")
class Mvlgamma(UnaryMathOp):
    """Compute the multivariate log-gamma function with dimension p element-wise."""

    op_name: object = "Mvlgamma"


@register_op("SpecialGamma")
class SpecialGamma(UnaryMathOp):
    """Compute the gamma function element-wise."""

    op_name: object = "SpecialGamma"
    np_op_name: object = "gamma"


@register_op("BesselI0")
class BesselI0(UnaryMathOp):
    """Modify Bessel function of order 0."""

    op_name: object = "BesselI0"
    np_op_name: object = "i0"


@register_op("BesselI1")
class BesselI1(UnaryMathOp):
    """Modify Bessel function of order 1."""

    op_name: object = "BesselI1"


@register_op("Erfcinv")
class Erfcinv(UnaryMathOp):
    """Inverse complementary error function."""

    op_name: object = "Erfcinv"


@register_op("Ndtri")
class Ndtri(UnaryMathOp):
    """Inverse standard normal CDF."""

    op_name: object = "Ndtri"


@register_op("Lbeta")
class Lbeta(OpDef):
    """Log of the absolute value of the beta function."""

    op_name: object = "Lbeta"

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The x parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape: object = getattr(x, "shape", ())
        if len(shape) > 0:
            return shape[:-1]
        return ()


@register_op("BesselJ0")
class BesselJ0(UnaryMathOp):
    """BesselJ0 operation."""

    op_name: object = "BesselJ0"


@register_op("BesselJ1")
class BesselJ1(UnaryMathOp):
    """BesselJ1 operation."""

    op_name: object = "BesselJ1"


@register_op("BesselK0")
class BesselK0(UnaryMathOp):
    """BesselK0 operation."""

    op_name: object = "BesselK0"


@register_op("BesselK0e")
class BesselK0e(UnaryMathOp):
    """BesselK0e operation."""

    op_name: object = "BesselK0e"


@register_op("BesselK1")
class BesselK1(UnaryMathOp):
    """BesselK1 operation."""

    op_name: object = "BesselK1"


@register_op("BesselK1e")
class BesselK1e(UnaryMathOp):
    """BesselK1e operation."""

    op_name: object = "BesselK1e"


@register_op("BesselY0")
class BesselY0(UnaryMathOp):
    """BesselY0 operation."""

    op_name: object = "BesselY0"


@register_op("BesselY1")
class BesselY1(UnaryMathOp):
    """BesselY1 operation."""

    op_name: object = "BesselY1"


@register_op("Dawsn")
class Dawsn(UnaryMathOp):
    """Dawsn operation."""

    op_name: object = "Dawsn"


@register_op("Expint")
class Expint(UnaryMathOp):
    """Expint operation."""

    op_name: object = "Expint"


@register_op("FresnelCos")
class FresnelCos(UnaryMathOp):
    """FresnelCos operation."""

    op_name: object = "FresnelCos"


@register_op("FresnelSin")
class FresnelSin(UnaryMathOp):
    """FresnelSin operation."""

    op_name: object = "FresnelSin"


@register_op("Spence")
class Spence(UnaryMathOp):
    """Spence operation."""

    op_name: object = "Spence"


@register_op("ModifiedBesselI0")
class ModifiedBesselI0(UnaryMathOp):
    """ModifiedBesselI0 operation."""

    op_name: object = "ModifiedBesselI0"


@register_op("ModifiedBesselI1")
class ModifiedBesselI1(UnaryMathOp):
    """ModifiedBesselI1 operation."""

    op_name: object = "ModifiedBesselI1"


@register_op("ModifiedBesselK0")
class ModifiedBesselK0(UnaryMathOp):
    """ModifiedBesselK0 operation."""

    op_name: object = "ModifiedBesselK0"


@register_op("ModifiedBesselK1")
class ModifiedBesselK1(UnaryMathOp):
    """ModifiedBesselK1 operation."""

    op_name: object = "ModifiedBesselK1"
