"""Backend utilities."""

from typing import Any, Callable

import jax.ops
import jax.scipy.linalg
import jax.scipy.signal
import jax.scipy.special
import jax.scipy.special as jss
import jax.scipy.stats


def _execute_binom_cdf(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    k, n, p = args[0], args[1], args[2]
    loc = kwargs.get("loc", 0.0)
    return jss.betainc(n - (k - loc), (k - loc) + 1, 1 - p)


def _execute_bessel_jn(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return jax.scipy.special.bessel_jn(args[1], v=args[0])


def _execute_unsorted_segment_sum(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return jax.ops.segment_sum(*args, **kwargs)


def _execute_unsorted_segment_max(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return jax.ops.segment_max(*args, **kwargs)


def _execute_unsorted_segment_min(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return jax.ops.segment_min(*args, **kwargs)


def _execute_unsorted_segment_prod(*args: object, **kwargs: object) -> object:
    """Function docstring."""
    return jax.ops.segment_prod(*args, **kwargs)


_OP_DISPATCH: dict[str, Callable[..., Any]] = {
    "Convolve2d": jax.scipy.signal.convolve2d,
    "Fftconvolve": jax.scipy.signal.fftconvolve,
    "Welch": jax.scipy.signal.welch,
    "Convolve": jax.scipy.signal.convolve,
    "NormPdf": jax.scipy.stats.norm.pdf,
    "NormCdf": jax.scipy.stats.norm.cdf,
    "GammaPdf": jax.scipy.stats.gamma.pdf,
    "GammaCdf": jax.scipy.stats.gamma.cdf,
    "BetaPdf": jax.scipy.stats.beta.pdf,
    "BetaCdf": jax.scipy.stats.beta.cdf,
    "PoissonPmf": jax.scipy.stats.poisson.pmf,
    "PoissonCdf": jax.scipy.stats.poisson.cdf,
    "BinomPmf": jax.scipy.stats.binom.pmf,
    "BinomCdf": _execute_binom_cdf,
    "Erf": jax.scipy.special.erf,
    "SpecialGamma": jax.scipy.special.gamma,
    "BesselJn": _execute_bessel_jn,
    "Digamma": jax.scipy.special.digamma,
    "Polygamma": jax.scipy.special.polygamma,
    "Zeta": jax.scipy.special.zeta,
    "MatrixExponential": jax.scipy.linalg.expm,
    "Polar": jax.scipy.linalg.polar,
    "Schur": jax.scipy.linalg.schur,
    "SegmentSum": jax.ops.segment_sum,
    "SegmentMax": jax.ops.segment_max,
    "SegmentMin": jax.ops.segment_min,
    "SegmentProd": jax.ops.segment_prod,
    "UnsortedSegmentSum": _execute_unsorted_segment_sum,
    "UnsortedSegmentMax": _execute_unsorted_segment_max,
    "UnsortedSegmentMin": _execute_unsorted_segment_min,
    "UnsortedSegmentProd": _execute_unsorted_segment_prod,
}


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (type): The cls parameter for the operation.
        op_type (str): Argument op_type.
        *args (object): Argument *args.
        **kwargs (object): Argument **kwargs.

    Returns:
        object: The result.
    """
    if op_type in _OP_DISPATCH:
        return _OP_DISPATCH[op_type](*args, **kwargs)
    raise NotImplementedError(f"Operation '{op_type}' not supported eagerly by this backend.")
