# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Generate random ops module."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .frontend import sobol_sample
from .sobol import SobolSample


@register_op("Binomial")
class Binomial(OpDef):
    """Operator for generating random numbers from a Binomial distribution."""

    op_name: object = "Binomial"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


@register_op("Cauchy")
class Cauchy(OpDef):
    """Operator for generating random numbers from a Cauchy distribution."""

    op_name: object = "Cauchy"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Chisquare")
class Chisquare(OpDef):
    """Operator for generating random numbers from a Chisquare distribution."""

    op_name: object = "Chisquare"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Dirichlet")
class Dirichlet(OpDef):
    """Operator for generating random numbers from a Dirichlet distribution."""

    op_name: object = "Dirichlet"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("DoubleSidedMaxwell")
class DoubleSidedMaxwell(OpDef):
    """Operator for generating random numbers from a double-sided Maxwell distribution."""

    op_name: object = "DoubleSidedMaxwell"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Exponential")
class Exponential(OpDef):
    """Operator for generating random numbers from an Exponential distribution."""

    op_name: object = "Exponential"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("F")
class F(OpDef):
    """Operator for generating random numbers from a F distribution."""

    op_name: object = "F"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Gumbel")
class Gumbel(OpDef):
    """Operator for generating random numbers from a Gumbel distribution."""

    op_name: object = "Gumbel"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Laplace")
class Laplace(OpDef):
    """Operator for generating random numbers from a Laplace distribution."""

    op_name: object = "Laplace"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Loggamma")
class Loggamma(OpDef):
    """Operator for generating random numbers from a Loggamma distribution."""

    op_name: object = "Loggamma"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Logistic")
class Logistic(OpDef):
    """Operator for generating random numbers from a Logistic distribution."""

    op_name: object = "Logistic"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Lognormal")
class Lognormal(OpDef):
    """Operator for generating random numbers from a Lognormal distribution."""

    op_name: object = "Lognormal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Maxwell")
class Maxwell(OpDef):
    """Operator for generating random numbers from a Maxwell distribution."""

    op_name: object = "Maxwell"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("MultivariateNormal")
class MultivariateNormal(OpDef):
    """Operator for generating random vectors from a multivariate normal distribution."""

    op_name: object = "MultivariateNormal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Pareto")
class Pareto(OpDef):
    """Operator for generating random numbers from a Pareto distribution."""

    op_name: object = "Pareto"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Poisson")
class Poisson(OpDef):
    """Operator for generating random numbers from a Poisson distribution."""

    op_name: object = "Poisson"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Rayleigh")
class Rayleigh(OpDef):
    """Operator for generating random numbers from a Rayleigh distribution."""

    op_name: object = "Rayleigh"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("T")
class T(OpDef):
    """Operator for generating random numbers from a T distribution."""

    op_name: object = "T"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Triangular")
class Triangular(OpDef):
    """Operator for generating random numbers from a Triangular distribution."""

    op_name: object = "Triangular"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Wald")
class Wald(OpDef):
    """Operator for generating random numbers from a Wald distribution."""

    op_name: object = "Wald"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("WeibullMin")
class WeibullMin(OpDef):
    """Operator for generating random numbers from a Weibull minimum extreme value distribution."""

    op_name: object = "WeibullMin"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Clone")
class Clone(OpDef):
    """Operator for cloning pseudo-random number generator keys."""

    op_name: object = "Clone"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("KeyData")
class KeyData(OpDef):
    """Operator representing the underlying state data of a pseudo-random key."""

    op_name: object = "KeyData"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("KeyImpl")
class KeyImpl(OpDef):
    """Operator representing the implementation details of a pseudo-random key."""

    op_name: object = "KeyImpl"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("WrapKeyData")
class WrapKeyData(OpDef):
    """Operator for wrapping raw state data into a pseudo-random key format."""

    op_name: object = "WrapKeyData"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Bits")
class Bits(OpDef):
    """Operator for generating raw pseudo-random bits."""

    op_name: object = "Bits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("GeneralizedNormal")
class GeneralizedNormal(OpDef):
    """Operator for generating random numbers from a generalized normal distribution."""

    op_name: object = "GeneralizedNormal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Orthogonal")
class Orthogonal(OpDef):
    """Operator for generating random numbers from an Orthogonal distribution."""

    op_name: object = "Orthogonal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("RandomGammaP")
class RandomGammaP(OpDef):
    """Operator for generating random numbers from a Gamma distribution using the 'P' parameterization."""

    op_name: object = "RandomGammaP"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Ball")
class Ball(OpDef):
    """Operator for generating random points uniformly distributed within a multi-dimensional ball."""

    op_name: object = "Ball"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Key")
class Key(OpDef):
    """Operator for instantiating or manipulating pseudo-random number generator keys."""

    op_name: object = "Key"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("RngBitGenerator")
class RngBitGenerator(OpDef):
    """Operator for generating random bits using a specified pseudo-random number generator algorithm."""

    op_name: object = "RngBitGenerator"

    def infer_shape(self, key: object, shape: object, dtype: object, **kwargs: object) -> object:
        """Infer the output shape for the RngBitGenerator operation.

        Args:
            key (object): The PRNG key state.
            shape (object): The desired output shape.
            dtype (object): The desired output data type.
            **kwargs (object): Additional keyword arguments.

        Returns: object: The computed output shape, which matches the input shape parameter.
        """
        return shape


@register_op("RngUniform")
class RngUniform(OpDef):
    """Operator for generating uniformly distributed random numbers within a specified range."""

    op_name: object = "RngUniform"

    def infer_shape(self, a: object, b: object, shape: object, dtype: object, **kwargs: object) -> object:
        """Infer the output shape for the RngUniform operation.

        Args:
            a (object): The lower bound of the uniform distribution.
            b (object): The upper bound of the uniform distribution.
            shape (object): The desired output shape.
            dtype (object): The desired output data type.
            **kwargs (object): Additional keyword arguments.

        Returns: object: The computed output shape, which matches the input shape parameter.
        """
        return shape


@register_op("Beta")
class Beta(OpDef):
    """Operator for generating random numbers from a Beta distribution."""

    op_name: object = "Beta"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


@register_op("Gamma")
class Gamma(OpDef):
    """Operator for generating random numbers from a Gamma distribution."""

    op_name: object = "Gamma"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape based on the provided inputs.

        Args:
            *args (object): Positional arguments typically representing input shapes or values.
            **kwargs (object): Keyword arguments representing additional configurations.

        Returns: object: The evaluated shape for this operation, usually derived from the first argument.
        """
        return args[0] if args else ()


__all__ = [
    "SobolSample",
    "sobol_sample",
]


@register_op("categorical")
class categorical(OpDef):
    """categorical operation."""

    op_name: object = "categorical"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("dirichlet")
class dirichlet(OpDef):
    """dirichlet operation."""

    op_name: object = "dirichlet"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("binomial")
class binomial(OpDef):
    """binomial operation."""

    op_name: object = "binomial"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("truncated_normal")
class truncated_normal(OpDef):
    """truncated_normal operation."""

    op_name: object = "truncated_normal"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("permutation")
class permutation(OpDef):
    """permutation operation."""

    op_name: object = "permutation"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("choice")
class choice(OpDef):
    """choice operation."""

    op_name: object = "choice"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(inputs, "shape", ())
