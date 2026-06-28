"""Backend utilities."""

import jax.numpy as jnp
import jax.ops
import jax.scipy.linalg
import jax.scipy.special

import jax.scipy.stats

import jax.scipy.signal

from ml_switcheroo_compiler.backends.eager import execute_generic_op


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:  # noqa: C901, PLR0911, PLR0912
    """Execute execute_op.

    Args:
        cls (Any): The cls parameter for the operation.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    if op_type == "Convolve2d":
        return jax.scipy.signal.convolve2d(*args, **kwargs)
    if op_type == "Fftconvolve":
        return jax.scipy.signal.fftconvolve(*args, **kwargs)
    if op_type == "Welch":
        return jax.scipy.signal.welch(*args, **kwargs)
    if op_type == "Convolve":
        return jax.scipy.signal.convolve(*args, **kwargs)
    if op_type == "NormPdf":
        return jax.scipy.stats.norm.pdf(*args, **kwargs)
    if op_type == "NormCdf":
        return jax.scipy.stats.norm.cdf(*args, **kwargs)
    if op_type == "GammaPdf":
        return jax.scipy.stats.gamma.pdf(*args, **kwargs)
    if op_type == "GammaCdf":
        return jax.scipy.stats.gamma.cdf(*args, **kwargs)
    if op_type == "BetaPdf":
        return jax.scipy.stats.beta.pdf(*args, **kwargs)
    if op_type == "BetaCdf":
        return jax.scipy.stats.beta.cdf(*args, **kwargs)
    if op_type == "PoissonPmf":
        return jax.scipy.stats.poisson.pmf(*args, **kwargs)
    if op_type == "PoissonCdf":
        return jax.scipy.stats.poisson.cdf(*args, **kwargs)
    if op_type == "BinomPmf":
        return jax.scipy.stats.binom.pmf(*args, **kwargs)
    if op_type == "BinomCdf":
        import jax.scipy.special as jss

        # JAX doesn't have binom.cdf natively in scipy.stats.binom
        k, n, p, loc = args[0], args[1], args[2], kwargs.get("loc", 0.0)
        return jss.betainc(n - (k - loc), (k - loc) + 1, 1 - p)

    if op_type == "Erf":
        return jax.scipy.special.erf(*args, **kwargs)
    if op_type == "SpecialGamma":
        return jax.scipy.special.gamma(*args, **kwargs)
    if op_type == "BesselJn":
        return jax.scipy.special.bessel_jn(args[1], v=args[0])
    if op_type == "Digamma":
        return jax.scipy.special.digamma(*args, **kwargs)
    if op_type == "Polygamma":
        return jax.scipy.special.polygamma(*args, **kwargs)
    if op_type == "Zeta":
        return jax.scipy.special.zeta(*args, **kwargs)
    if op_type == "MatrixExponential":
        return jax.scipy.linalg.expm(*args, **kwargs)
    if op_type == "Polar":
        return jax.scipy.linalg.polar(*args, **kwargs)
    if op_type == "Schur":
        return jax.scipy.linalg.schur(*args, **kwargs)
    if op_type == "SegmentSum":
        return jax.ops.segment_sum(*args, **kwargs)
    if op_type == "SegmentMax":
        return jax.ops.segment_max(*args, **kwargs)
    if op_type == "SegmentMin":
        return jax.ops.segment_min(*args, **kwargs)
    if op_type == "SegmentProd":
        return jax.ops.segment_prod(*args, **kwargs)
    if op_type in (
        "UnsortedSegmentSum",
        "UnsortedSegmentMax",
        "UnsortedSegmentMin",
        "UnsortedSegmentProd",
    ):
        op_name = op_type.replace("UnsortedSegment", "segment_").lower()
        return getattr(jax.ops, op_name)(*args, **kwargs)

    return execute_generic_op(jnp, op_type, *args, **kwargs)
