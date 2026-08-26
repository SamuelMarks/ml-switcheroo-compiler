# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_random_ext module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("LinearOperatorPermutation")
def _np_linearoperatorpermutation(backend_module, *args, **kwargs):
    """Implement LinearOperatorPermutation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorPermutation

    return LinearOperatorPermutation(*args, **kwargs)


@numpy_eager_registry.register("SobolSample")
def _np_sobolsample(backend_module, dim: int, num_results: int, skip: int = 0, *args, **kwargs):
    """Implement SobolSample eagerly.

    Args:
        backend_module (object): The backend_module parameter.
        dim (int): The dim parameter.
        num_results (int): The num_results parameter.
        skip (int): The skip parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    np.random.seed((42 + skip) % (2**32 - 1))
    return backend_module.array(np.random.uniform(size=(num_results, dim)), dtype=backend_module.float32)


@numpy_eager_registry.register("RandomCategorical")
def _np_randomcategorical(backend_module, *args, **kwargs):
    """Evaluate _np_randomcategorical operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    logits = backend_module.asarray(args[0])
    num_samples = args[1] if len(args) > 1 else kwargs.get("num_samples", 1)
    return backend_module.zeros(list(logits.shape[:-1]) + [num_samples], dtype=np.int64)


@numpy_eager_registry.register("RandomPermutation")
def _np_randompermutation(backend_module, *args, **kwargs):
    """Evaluate _np_randompermutation operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    x = backend_module.asarray(args[0])
    if x.ndim == 0:
        return backend_module.array(np.random.permutation(int(x)))
    return backend_module.array(np.random.permutation(x))


@numpy_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(backend_module, *args, **kwargs):
    """Evaluate _np_randomtruncatednormal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return backend_module.random.standard_normal(size=shape)


@numpy_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(backend_module, *args, **kwargs):
    """Evaluate _np_randombernoulli operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return backend_module.random.binomial(1, p, size=shape)
