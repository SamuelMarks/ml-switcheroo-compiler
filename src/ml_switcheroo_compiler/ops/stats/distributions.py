# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Statistical distributions."""

from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op


@register_op("NormPdf")
class NormPdf(OpDef):
    """Operation for calculating the probability density function (PDF) of a normal distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the normal PDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `x`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `x`.
        """
        return args[0].shape


@register_op("NormCdf")
class NormCdf(OpDef):
    """Operation for calculating the cumulative distribution function (CDF) of a normal distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the normal CDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `x`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `x`.
        """
        return args[0].shape


@register_op("GammaPdf")
class GammaPdf(OpDef):
    """Operation for calculating the probability density function (PDF) of a gamma distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the gamma PDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `x`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `x`.
        """
        return args[0].shape


@register_op("GammaCdf")
class GammaCdf(OpDef):
    """Operation for calculating the cumulative distribution function (CDF) of a gamma distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the gamma CDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `x`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `x`.
        """
        return args[0].shape


@register_op("BetaPdf")
class BetaPdf(OpDef):
    """Operation for calculating the probability density function (PDF) of a beta distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the beta PDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `x`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `x`.
        """
        return args[0].shape


@register_op("BetaCdf")
class BetaCdf(OpDef):
    """Operation for calculating the cumulative distribution function (CDF) of a beta distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the beta CDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `x`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `x`.
        """
        return args[0].shape


@register_op("PoissonPmf")
class PoissonPmf(OpDef):
    """Operation for calculating the probability mass function (PMF) of a Poisson distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the Poisson PMF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `k`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `k`.
        """
        return args[0].shape


@register_op("PoissonCdf")
class PoissonCdf(OpDef):
    """Operation for calculating the cumulative distribution function (CDF) of a Poisson distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the Poisson CDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `k`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `k`.
        """
        return args[0].shape


@register_op("BinomPmf")
class BinomPmf(OpDef):
    """Operation for calculating the probability mass function (PMF) of a binomial distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the binomial PMF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `k`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `k`.
        """
        return args[0].shape


@register_op("BinomCdf")
class BinomCdf(OpDef):
    """Operation for calculating the cumulative distribution function (CDF) of a binomial distribution."""

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the binomial CDF operation.

        Args:
            *args: Positional arguments, where the first argument is the input tensor `k`.
            **kwargs: Keyword arguments.

        Returns: Tensor: The inferred shape, matching the shape of the input tensor `k`.
        """
        return args[0].shape


def norm_pdf(x, loc: float = 0.0, scale: float = 1.0):
    """Calculate the probability density function (PDF) for a normal distribution.

    Args:
        x: The input values at which to evaluate the normal PDF.
        loc: The mean (center) of the normal distribution.
        scale: The standard deviation (spread) of the normal distribution.

    Returns: Tensor: A tensor containing the evaluated normal PDF values.
    """
    return get_op("NormPdf")()(x, loc, scale)


def norm_cdf(x, loc: float = 0.0, scale: float = 1.0):
    """Calculate the cumulative distribution function (CDF) for a normal distribution.

    Args:
        x: The input values at which to evaluate the normal CDF.
        loc: The mean (center) of the normal distribution.
        scale: The standard deviation (spread) of the normal distribution.

    Returns: Tensor: A tensor containing the evaluated normal CDF values.
    """
    return get_op("NormCdf")()(x, loc, scale)


def gamma_pdf(x, a, loc: float = 0.0, scale: float = 1.0):
    """Calculate the probability density function (PDF) for a gamma distribution.

    Args:
        x: The input values at which to evaluate the gamma PDF.
        a: The shape parameter of the gamma distribution.
        loc: The location parameter (shift) of the gamma distribution.
        scale: The scale parameter of the gamma distribution.

    Returns: Tensor: A tensor containing the evaluated gamma PDF values.
    """
    return get_op("GammaPdf")()(x, a, loc, scale)


def gamma_cdf(x, a, loc: float = 0.0, scale: float = 1.0):
    """Calculate the cumulative distribution function (CDF) for a gamma distribution.

    Args:
        x: The input values at which to evaluate the gamma CDF.
        a: The shape parameter of the gamma distribution.
        loc: The location parameter (shift) of the gamma distribution.
        scale: The scale parameter of the gamma distribution.

    Returns: Tensor: A tensor containing the evaluated gamma CDF values.
    """
    return get_op("GammaCdf")()(x, a, loc, scale)


def beta_pdf(x, a, b, loc: float = 0.0, scale: float = 1.0):
    """Calculate the probability density function (PDF) for a beta distribution.

    Args:
        x: The input values at which to evaluate the beta PDF.
        a: The first shape parameter (alpha) of the beta distribution.
        b: The second shape parameter (beta) of the beta distribution.
        loc: The location parameter (shift) of the beta distribution.
        scale: The scale parameter of the beta distribution.

    Returns: Tensor: A tensor containing the evaluated beta PDF values.
    """
    return get_op("BetaPdf")()(x, a, b, loc, scale)


def beta_cdf(x, a, b, loc: float = 0.0, scale: float = 1.0):
    """Calculate the cumulative distribution function (CDF) for a beta distribution.

    Args:
        x: The input values at which to evaluate the beta CDF.
        a: The first shape parameter (alpha) of the beta distribution.
        b: The second shape parameter (beta) of the beta distribution.
        loc: The location parameter (shift) of the beta distribution.
        scale: The scale parameter of the beta distribution.

    Returns: Tensor: A tensor containing the evaluated beta CDF values.
    """
    return get_op("BetaCdf")()(x, a, b, loc, scale)


def poisson_pmf(k, mu, loc: float = 0.0):
    """Calculate the probability mass function (PMF) for a Poisson distribution.

    Args:
        k: The input values at which to evaluate the Poisson PMF.
        mu: The expected number of events (lambda/rate parameter).
        loc: The location parameter (shift) of the Poisson distribution.

    Returns: Tensor: A tensor containing the evaluated Poisson PMF values.
    """
    return get_op("PoissonPmf")()(k, mu, loc)


def poisson_cdf(k, mu, loc: float = 0.0):
    """Calculate the cumulative distribution function (CDF) for a Poisson distribution.

    Args:
        k: The input values at which to evaluate the Poisson CDF.
        mu: The expected number of events (lambda/rate parameter).
        loc: The location parameter (shift) of the Poisson distribution.

    Returns: Tensor: A tensor containing the evaluated Poisson CDF values.
    """
    return get_op("PoissonCdf")()(k, mu, loc)


def binom_pmf(k, n, p, loc: float = 0.0):
    """Calculate the probability mass function (PMF) for a binomial distribution.

    Args:
        k: The input values at which to evaluate the binomial PMF.
        n: The number of trials.
        p: The probability of success for each trial.
        loc: The location parameter (shift) of the binomial distribution.

    Returns: Tensor: A tensor containing the evaluated binomial PMF values.
    """
    return get_op("BinomPmf")()(k, n, p, loc)


def binom_cdf(k, n, p, loc: float = 0.0):
    """Calculate the cumulative distribution function (CDF) for a binomial distribution.

    Args:
        k: The input values at which to evaluate the binomial CDF.
        n: The number of trials.
        p: The probability of success for each trial.
        loc: The location parameter (shift) of the binomial distribution.

    Returns: Tensor: A tensor containing the evaluated binomial CDF values.
    """
    return get_op("BinomCdf")()(k, n, p, loc)


__all__ = [
    "beta_cdf",
    "beta_pdf",
    "binom_cdf",
    "binom_pmf",
    "gamma_cdf",
    "gamma_pdf",
    "norm_cdf",
    "norm_pdf",
    "poisson_cdf",
    "poisson_pmf",
]
