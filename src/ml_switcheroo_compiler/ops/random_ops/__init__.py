# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Generate random ops module."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .frontend import sobol_sample
from .sobol import SobolSample


@register_op("Binomial")
class Binomial(OpDef):
    """Operator for generating random numbers from a Binomial distribution."""

    op_name = "Binomial"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return args[0] if args else ()


@register_op("Cauchy")
class Cauchy(OpDef):
    """Operator for generating random numbers from a Cauchy distribution."""

    op_name = "Cauchy"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Chisquare")
class Chisquare(OpDef):
    """Operator for generating random numbers from a Chisquare distribution."""

    op_name = "Chisquare"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Dirichlet")
class Dirichlet(OpDef):
    """Operator for generating random numbers from a Dirichlet distribution."""

    op_name = "Dirichlet"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("DoubleSidedMaxwell")
class DoubleSidedMaxwell(OpDef):
    """Operator for generating random numbers from a double-sided Maxwell distribution."""

    op_name = "DoubleSidedMaxwell"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Exponential")
class Exponential(OpDef):
    """Operator for generating random numbers from an Exponential distribution."""

    op_name = "Exponential"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("F")
class F(OpDef):
    """Operator for generating random numbers from a F distribution."""

    op_name = "F"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Gumbel")
class Gumbel(OpDef):
    """Operator for generating random numbers from a Gumbel distribution."""

    op_name = "Gumbel"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Laplace")
class Laplace(OpDef):
    """Operator for generating random numbers from a Laplace distribution."""

    op_name = "Laplace"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Loggamma")
class Loggamma(OpDef):
    """Operator for generating random numbers from a Loggamma distribution."""

    op_name = "Loggamma"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Logistic")
class Logistic(OpDef):
    """Operator for generating random numbers from a Logistic distribution."""

    op_name = "Logistic"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Lognormal")
class Lognormal(OpDef):
    """Operator for generating random numbers from a Lognormal distribution."""

    op_name = "Lognormal"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Maxwell")
class Maxwell(OpDef):
    """Operator for generating random numbers from a Maxwell distribution."""

    op_name = "Maxwell"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("MultivariateNormal")
class MultivariateNormal(OpDef):
    """Operator for generating random vectors from a multivariate normal distribution."""

    op_name = "MultivariateNormal"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Pareto")
class Pareto(OpDef):
    """Operator for generating random numbers from a Pareto distribution."""

    op_name = "Pareto"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Poisson")
class Poisson(OpDef):
    """Operator for generating random numbers from a Poisson distribution."""

    op_name = "Poisson"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Rayleigh")
class Rayleigh(OpDef):
    """Operator for generating random numbers from a Rayleigh distribution."""

    op_name = "Rayleigh"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("T")
class T(OpDef):
    """Operator for generating random numbers from a T distribution."""

    op_name = "T"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Triangular")
class Triangular(OpDef):
    """Operator for generating random numbers from a Triangular distribution."""

    op_name = "Triangular"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Wald")
class Wald(OpDef):
    """Operator for generating random numbers from a Wald distribution."""

    op_name = "Wald"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("WeibullMin")
class WeibullMin(OpDef):
    """Operator for generating random numbers from a Weibull minimum extreme value distribution."""

    op_name = "WeibullMin"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Clone")
class Clone(OpDef):
    """Operator for cloning pseudo-random number generator keys."""

    op_name = "Clone"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("KeyData")
class KeyData(OpDef):
    """Operator representing the underlying state data of a pseudo-random key."""

    op_name = "KeyData"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("KeyImpl")
class KeyImpl(OpDef):
    """Operator representing the implementation details of a pseudo-random key."""

    op_name = "KeyImpl"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("WrapKeyData")
class WrapKeyData(OpDef):
    """Operator for wrapping raw state data into a pseudo-random key format."""

    op_name = "WrapKeyData"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Bits")
class Bits(OpDef):
    """Operator for generating raw pseudo-random bits."""

    op_name = "Bits"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("GeneralizedNormal")
class GeneralizedNormal(OpDef):
    """Operator for generating random numbers from a generalized normal distribution."""

    op_name = "GeneralizedNormal"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Orthogonal")
class Orthogonal(OpDef):
    """Operator for generating random numbers from an Orthogonal distribution."""

    op_name = "Orthogonal"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("RandomGammaP")
class RandomGammaP(OpDef):
    """Operator for generating random numbers from a Gamma distribution using the 'P' parameterization."""

    op_name = "RandomGammaP"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Ball")
class Ball(OpDef):
    """Operator for generating random points uniformly distributed within a multi-dimensional ball."""

    op_name = "Ball"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Key")
class Key(OpDef):
    """Operator for instantiating or manipulating pseudo-random number generator keys."""

    op_name = "Key"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("RngBitGenerator")
class RngBitGenerator(OpDef):
    """Operator for generating random bits using a specified pseudo-random number generator algorithm."""

    op_name = "RngBitGenerator"

    def infer_shape(self, key: Any, shape: Any, dtype: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the RngBitGenerator operation.

        Args:
            key (object): The PRNG key state.
            shape (object): The desired output shape.
            dtype (object): The desired output data type.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The computed output shape, which matches the input shape parameter.
        """
        return shape


@register_op("RngUniform")
class RngUniform(OpDef):
    """Operator for generating uniformly distributed random numbers within a specified range."""

    op_name = "RngUniform"

    def infer_shape(self, a: Any, b: Any, shape: Any, dtype: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the RngUniform operation.

        Args:
            a (object): The lower bound of the uniform distribution.
            b (object): The upper bound of the uniform distribution.
            shape (object): The desired output shape.
            dtype (object): The desired output data type.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The computed output shape, which matches the input shape parameter.
        """
        return shape


@register_op("Beta")
class Beta(OpDef):
    """Operator for generating random numbers from a Beta distribution."""

    op_name = "Beta"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Gamma")
class Gamma(OpDef):
    """Operator for generating random numbers from a Gamma distribution."""

    op_name = "Gamma"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: Any: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


__all__ = [
    "SobolSample",
    "sobol_sample",
]


@register_op("categorical")
class categorical(OpDef):
    """categorical operation."""

    op_name = "categorical"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("dirichlet")
class dirichlet(OpDef):
    """dirichlet operation."""

    op_name = "dirichlet"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("binomial")
class binomial(OpDef):
    """binomial operation."""

    op_name = "binomial"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("truncated_normal")
class truncated_normal(OpDef):
    """truncated_normal operation."""

    op_name = "truncated_normal"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("permutation")
class permutation(OpDef):
    """permutation operation."""

    op_name = "permutation"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("choice")
class choice(OpDef):
    """choice operation."""

    op_name = "choice"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())
