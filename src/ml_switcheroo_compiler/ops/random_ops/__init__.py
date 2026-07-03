"""Random ops module."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .frontend import sobol_sample
from .sobol import SobolSample


@register_op("Binomial")
class Binomial(OpDef):
    """Binomial operator definition."""

    op_name = "Binomial"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Cauchy")
class Cauchy(OpDef):
    """Cauchy operator definition."""

    op_name = "Cauchy"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Chisquare")
class Chisquare(OpDef):
    """Chisquare operator definition."""

    op_name = "Chisquare"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Dirichlet")
class Dirichlet(OpDef):
    """Dirichlet operator definition."""

    op_name = "Dirichlet"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("DoubleSidedMaxwell")
class DoubleSidedMaxwell(OpDef):
    """DoubleSidedMaxwell operator definition."""

    op_name = "DoubleSidedMaxwell"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Exponential")
class Exponential(OpDef):
    """Exponential operator definition."""

    op_name = "Exponential"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("F")
class F(OpDef):
    """F operator definition."""

    op_name = "F"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Gumbel")
class Gumbel(OpDef):
    """Gumbel operator definition."""

    op_name = "Gumbel"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Laplace")
class Laplace(OpDef):
    """Laplace operator definition."""

    op_name = "Laplace"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Loggamma")
class Loggamma(OpDef):
    """Loggamma operator definition."""

    op_name = "Loggamma"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Logistic")
class Logistic(OpDef):
    """Logistic operator definition."""

    op_name = "Logistic"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Lognormal")
class Lognormal(OpDef):
    """Lognormal operator definition."""

    op_name = "Lognormal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Maxwell")
class Maxwell(OpDef):
    """Maxwell operator definition."""

    op_name = "Maxwell"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("MultivariateNormal")
class MultivariateNormal(OpDef):
    """MultivariateNormal operator definition."""

    op_name = "MultivariateNormal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Pareto")
class Pareto(OpDef):
    """Pareto operator definition."""

    op_name = "Pareto"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Poisson")
class Poisson(OpDef):
    """Poisson operator definition."""

    op_name = "Poisson"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Rayleigh")
class Rayleigh(OpDef):
    """Rayleigh operator definition."""

    op_name = "Rayleigh"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("T")
class T(OpDef):
    """T operator definition."""

    op_name = "T"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Triangular")
class Triangular(OpDef):
    """Triangular operator definition."""

    op_name = "Triangular"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Wald")
class Wald(OpDef):
    """Wald operator definition."""

    op_name = "Wald"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("WeibullMin")
class WeibullMin(OpDef):
    """WeibullMin operator definition."""

    op_name = "WeibullMin"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Clone")
class Clone(OpDef):
    """Clone operator definition."""

    op_name = "Clone"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("KeyData")
class KeyData(OpDef):
    """KeyData operator definition."""

    op_name = "KeyData"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("KeyImpl")
class KeyImpl(OpDef):
    """KeyImpl operator definition."""

    op_name = "KeyImpl"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("WrapKeyData")
class WrapKeyData(OpDef):
    """WrapKeyData operator definition."""

    op_name = "WrapKeyData"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Bits")
class Bits(OpDef):
    """Bits operator definition."""

    op_name = "Bits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("GeneralizedNormal")
class GeneralizedNormal(OpDef):
    """GeneralizedNormal operator definition."""

    op_name = "GeneralizedNormal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Orthogonal")
class Orthogonal(OpDef):
    """Orthogonal operator definition."""

    op_name = "Orthogonal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("RandomGammaP")
class RandomGammaP(OpDef):
    """RandomGammaP operator definition."""

    op_name = "RandomGammaP"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0] if args else ()


@register_op("Ball")
class Ball(OpDef):
    """Ball operator definition."""

    op_name = "Ball"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        return args[0] if args else ()


@register_op("Key")
class Key(OpDef):
    """Key operator definition."""

    op_name = "Key"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        return args[0] if args else ()


@register_op("RngBitGenerator")
class RngBitGenerator(OpDef):
    """RngBitGenerator op."""

    op_name = "RngBitGenerator"

    def infer_shape(self, key: object, shape: object, dtype: object, **kwargs: object) -> object:
        """Function docstring."""
        return shape


@register_op("RngUniform")
class RngUniform(OpDef):
    """RngUniform op."""

    op_name = "RngUniform"

    def infer_shape(self, a: object, b: object, shape: object, dtype: object, **kwargs: object) -> object:
        """Function docstring."""
        return shape


@register_op("Beta")
class Beta(OpDef):
    """Beta operator definition."""

    op_name = "Beta"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        return args[0] if args else ()


@register_op("Gamma")
class Gamma(OpDef):
    """Gamma operator definition."""

    op_name = "Gamma"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        return args[0] if args else ()


__all__ = [
    "SobolSample",
    "sobol_sample",
]
