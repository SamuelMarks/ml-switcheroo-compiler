"""Statistical distributions."""

from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op


@register_op("NormPdf")
class NormPdf(OpDef):
    """NormPdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("NormCdf")
class NormCdf(OpDef):
    """NormCdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("GammaPdf")
class GammaPdf(OpDef):
    """GammaPdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("GammaCdf")
class GammaCdf(OpDef):
    """GammaCdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("BetaPdf")
class BetaPdf(OpDef):
    """BetaPdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("BetaCdf")
class BetaCdf(OpDef):
    """BetaCdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("PoissonPmf")
class PoissonPmf(OpDef):
    """PoissonPmf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("PoissonCdf")
class PoissonCdf(OpDef):
    """PoissonCdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("BinomPmf")
class BinomPmf(OpDef):
    """BinomPmf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


@register_op("BinomCdf")
class BinomCdf(OpDef):
    """BinomCdf."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape


def norm_pdf(x: object, loc: float = 0.0, scale: float = 1.0) -> object:
    """Evaluate norm_pdf."""
    return get_op("NormPdf")()(x, loc, scale)


def norm_cdf(x: object, loc: float = 0.0, scale: float = 1.0) -> object:
    """Evaluate norm_cdf."""
    return get_op("NormCdf")()(x, loc, scale)


def gamma_pdf(x: object, a: object, loc: float = 0.0, scale: float = 1.0) -> object:
    """Evaluate gamma_pdf."""
    return get_op("GammaPdf")()(x, a, loc, scale)


def gamma_cdf(x: object, a: object, loc: float = 0.0, scale: float = 1.0) -> object:
    """Evaluate gamma_cdf."""
    return get_op("GammaCdf")()(x, a, loc, scale)


def beta_pdf(x: object, a: object, b: object, loc: float = 0.0, scale: float = 1.0) -> object:
    """Evaluate beta_pdf."""
    return get_op("BetaPdf")()(x, a, b, loc, scale)


def beta_cdf(x: object, a: object, b: object, loc: float = 0.0, scale: float = 1.0) -> object:
    """Evaluate beta_cdf."""
    return get_op("BetaCdf")()(x, a, b, loc, scale)


def poisson_pmf(k: object, mu: object, loc: float = 0.0) -> object:
    """Evaluate poisson_pmf."""
    return get_op("PoissonPmf")()(k, mu, loc)


def poisson_cdf(k: object, mu: object, loc: float = 0.0) -> object:
    """Evaluate poisson_cdf."""
    return get_op("PoissonCdf")()(k, mu, loc)


def binom_pmf(k: object, n: object, p: object, loc: float = 0.0) -> object:
    """Evaluate binom_pmf."""
    return get_op("BinomPmf")()(k, n, p, loc)


def binom_cdf(k: object, n: object, p: object, loc: float = 0.0) -> object:
    """Evaluate binom_cdf."""
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
