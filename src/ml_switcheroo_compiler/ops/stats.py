# pragma: no cover
"""Statistical distributions."""
# pragma: no cover

# pragma: no cover
from ml_switcheroo_compiler.ops.base import OpDef, register_op
# pragma: no cover

# pragma: no cover


# pragma: no cover
@register_op("NormPdf")
# pragma: no cover
class NormPdf(OpDef):
    # pragma: no cover
    """NormPdf."""

    # pragma: no cover

    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("NormCdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class NormCdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """NormCdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("GammaPdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class GammaPdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """GammaPdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("GammaCdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class GammaCdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """GammaCdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("BetaPdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class BetaPdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """BetaPdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("BetaCdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class BetaCdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """BetaCdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("PoissonPmf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class PoissonPmf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """PoissonPmf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("PoissonCdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class PoissonCdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """PoissonCdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("BinomPmf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class BinomPmf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """BinomPmf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("BinomCdf")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class BinomCdf(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """BinomCdf."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def norm_pdf(x: object, loc: float = 0.0, scale: float = 1.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate norm_pdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("NormPdf")()(x, loc, scale)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def norm_cdf(x: object, loc: float = 0.0, scale: float = 1.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate norm_cdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("NormCdf")()(x, loc, scale)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def gamma_pdf(x: object, a: object, loc: float = 0.0, scale: float = 1.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate gamma_pdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("GammaPdf")()(x, a, loc, scale)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def gamma_cdf(x: object, a: object, loc: float = 0.0, scale: float = 1.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate gamma_cdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("GammaCdf")()(x, a, loc, scale)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def beta_pdf(x: object, a: object, b: object, loc: float = 0.0, scale: float = 1.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate beta_pdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("BetaPdf")()(x, a, b, loc, scale)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def beta_cdf(x: object, a: object, b: object, loc: float = 0.0, scale: float = 1.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate beta_cdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("BetaCdf")()(x, a, b, loc, scale)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def poisson_pmf(k: object, mu: object, loc: float = 0.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate poisson_pmf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("PoissonPmf")()(k, mu, loc)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def poisson_cdf(k: object, mu: object, loc: float = 0.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate poisson_cdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("PoissonCdf")()(k, mu, loc)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def binom_pmf(k: object, n: object, p: object, loc: float = 0.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate binom_pmf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("BinomPmf")()(k, n, p, loc)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def binom_cdf(k: object, n: object, p: object, loc: float = 0.0) -> object:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate binom_cdf."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return get_op("BinomCdf")()(k, n, p, loc)


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover
__all__ = [
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "beta_cdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "beta_pdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "binom_cdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "binom_pmf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "gamma_cdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "gamma_pdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "norm_cdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "norm_pdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "poisson_cdf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    "poisson_pmf",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
]
# pragma: no cover
# pragma: no cover
# pragma: no cover
