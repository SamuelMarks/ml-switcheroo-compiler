"""Special operations."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def bessel_i0e(*args: object, **kwargs: object) -> object:
    """Execute bessel_i0e."""
    return dispatch("lax", "bessel_i0e", *args, **kwargs)


def bessel_i0e_p(*args: object, **kwargs: object) -> object:
    """Execute bessel_i0e_p."""
    return dispatch("lax", "bessel_i0e_p", *args, **kwargs)


def bessel_i1e(*args: object, **kwargs: object) -> object:
    """Execute bessel_i1e."""
    return dispatch("lax", "bessel_i1e", *args, **kwargs)


def bessel_i1e_p(*args: object, **kwargs: object) -> object:
    """Execute bessel_i1e_p."""
    return dispatch("lax", "bessel_i1e_p", *args, **kwargs)


def betainc(*args: object, **kwargs: object) -> object:
    """Execute betainc."""
    return dispatch("lax", "betainc", *args, **kwargs)


def digamma_p(*args: object, **kwargs: object) -> object:
    """Execute digamma_p."""
    return dispatch("lax", "digamma_p", *args, **kwargs)


def erf_inv(*args: object, **kwargs: object) -> object:
    """Execute erf_inv."""
    return dispatch("lax", "erf_inv", *args, **kwargs)


def erf_inv_p(*args: object, **kwargs: object) -> object:
    """Execute erf_inv_p."""
    return dispatch("lax", "erf_inv_p", *args, **kwargs)


def erf_p(*args: object, **kwargs: object) -> object:
    """Execute erf_p."""
    return dispatch("lax", "erf_p", *args, **kwargs)


def erfc_p(*args: object, **kwargs: object) -> object:
    """Execute erfc_p."""
    return dispatch("lax", "erfc_p", *args, **kwargs)


def exp2_p(*args: object, **kwargs: object) -> object:
    """Execute exp2_p."""
    return dispatch("lax", "exp2_p", *args, **kwargs)


def exp_p(*args: object, **kwargs: object) -> object:
    """Execute exp_p."""
    return dispatch("lax", "exp_p", *args, **kwargs)


def expm1_p(*args: object, **kwargs: object) -> object:
    """Execute expm1_p."""
    return dispatch("lax", "expm1_p", *args, **kwargs)


def igamma(*args: object, **kwargs: object) -> object:
    """Execute igamma."""
    return dispatch("lax", "igamma", *args, **kwargs)


def igamma_grad_a(*args: object, **kwargs: object) -> object:
    """Execute igamma_grad_a."""
    return dispatch("lax", "igamma_grad_a", *args, **kwargs)


def igamma_grad_a_p(*args: object, **kwargs: object) -> object:
    """Execute igamma_grad_a_p."""
    return dispatch("lax", "igamma_grad_a_p", *args, **kwargs)


def igamma_p(*args: object, **kwargs: object) -> object:
    """Execute igamma_p."""
    return dispatch("lax", "igamma_p", *args, **kwargs)


def igammac(*args: object, **kwargs: object) -> object:
    """Execute igammac."""
    return dispatch("lax", "igammac", *args, **kwargs)


def igammac_p(*args: object, **kwargs: object) -> object:
    """Execute igammac_p."""
    return dispatch("lax", "igammac_p", *args, **kwargs)


def lgamma_p(*args: object, **kwargs: object) -> object:
    """Execute lgamma_p."""
    return dispatch("lax", "lgamma_p", *args, **kwargs)


def log1p_p(*args: object, **kwargs: object) -> object:
    """Execute log1p_p."""
    return dispatch("lax", "log1p_p", *args, **kwargs)


def log_p(*args: object, **kwargs: object) -> object:
    """Execute log_p."""
    return dispatch("lax", "log_p", *args, **kwargs)


def logistic(*args: object, **kwargs: object) -> object:
    """Execute logistic."""
    return dispatch("lax", "logistic", *args, **kwargs)


def logistic_p(*args: object, **kwargs: object) -> object:
    """Execute logistic_p."""
    return dispatch("lax", "logistic_p", *args, **kwargs)


def nextafter_p(*args: object, **kwargs: object) -> object:
    """Execute nextafter_p."""
    return dispatch("lax", "nextafter_p", *args, **kwargs)


def polygamma(*args: object, **kwargs: object) -> object:
    """Execute polygamma."""
    return dispatch("lax", "polygamma", *args, **kwargs)


def polygamma_p(*args: object, **kwargs: object) -> object:
    """Execute polygamma_p."""
    return dispatch("lax", "polygamma_p", *args, **kwargs)


def rsqrt_p(*args: object, **kwargs: object) -> object:
    """Execute rsqrt_p."""
    return dispatch("lax", "rsqrt_p", *args, **kwargs)


def sqrt_p(*args: object, **kwargs: object) -> object:
    """Execute sqrt_p."""
    return dispatch("lax", "sqrt_p", *args, **kwargs)


def zeta(*args: object, **kwargs: object) -> object:
    """Execute zeta."""
    return dispatch("lax", "zeta", *args, **kwargs)


def zeta_p(*args: object, **kwargs: object) -> object:
    """Execute zeta_p."""
    return dispatch("lax", "zeta_p", *args, **kwargs)
