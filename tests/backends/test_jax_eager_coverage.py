"""Module docstring."""

import jax.numpy as jnp

from ml_switcheroo_compiler.backends.jax.eager import execute_op
from ml_switcheroo_compiler.backends.jax.types import array, asarray, item, zeros


def test_jax_eager_ops() -> object:
    """Function docstring."""
    x = jnp.array([1.0, 2.0, 3.0])
    ids = jnp.array([0, 1, 0])

    execute_op(None, "SegmentSum", x, ids, num_segments=2)
    execute_op(None, "SegmentMax", x, ids, num_segments=2)
    execute_op(None, "SegmentMin", x, ids, num_segments=2)
    execute_op(None, "SegmentProd", x, ids, num_segments=2)
    execute_op(None, "UnsortedSegmentSum", x, ids, num_segments=2)
    execute_op(None, "UnsortedSegmentMax", x, ids, num_segments=2)
    execute_op(None, "UnsortedSegmentMin", x, ids, num_segments=2)
    execute_op(None, "UnsortedSegmentProd", x, ids, num_segments=2)

    mat = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    execute_op(None, "MatrixExponential", mat)
    execute_op(None, "Polar", mat)
    execute_op(None, "Schur", mat)

    execute_op(None, "Erf", x)
    execute_op(None, "SpecialGamma", x)
    execute_op(None, "Digamma", x)

    execute_op(None, "Polygamma", 1, x)
    execute_op(None, "Zeta", x, 2.0)

    execute_op(None, "NormPdf", x, loc=0.0, scale=1.0)
    execute_op(None, "NormCdf", x, loc=0.0, scale=1.0)
    execute_op(None, "GammaPdf", x, 1.0, loc=0.0, scale=1.0)
    execute_op(None, "GammaCdf", x, 1.0, loc=0.0, scale=1.0)
    execute_op(None, "BetaPdf", jnp.array([0.5]), 1.0, 1.0, loc=0.0, scale=1.0)
    execute_op(None, "BetaCdf", jnp.array([0.5]), 1.0, 1.0, loc=0.0, scale=1.0)

    execute_op(None, "PoissonPmf", jnp.array([1]), 1.0, loc=0.0)
    execute_op(None, "PoissonCdf", jnp.array([1]), 1.0, loc=0.0)

    execute_op(None, "BinomPmf", jnp.array([1]), 2, 0.5, loc=0.0)
    execute_op(None, "BinomCdf", jnp.array([1]), 2, 0.5, loc=0.0)

    x_1d = jnp.array([1.0, 2.0])
    execute_op(None, "Convolve", x_1d, x_1d)

    # scipy.special.bessel_jn takes v and z
    execute_op(None, "BesselJn", 1, x)

    x_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    execute_op(None, "Convolve2d", x_2d, x_2d)
    execute_op(None, "Fftconvolve", x_2d, x_2d)
    execute_op(None, "Welch", x_1d)

    # Fallback to generic execute
    try:
        execute_op(None, "UnknownFakeOp", x_1d)
    except NotImplementedError:
        pass

    assert zeros(None, (2,)) is not None
    assert array(None, [1, 2]) is not None
    assert asarray(None, [3, 4]) is not None
    assert item(None, jnp.array([5])) == 5
