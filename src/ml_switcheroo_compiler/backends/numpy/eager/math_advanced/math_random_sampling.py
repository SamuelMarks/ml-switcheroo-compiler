"""Implementation of math random sampling operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_random_sampling`)."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: F401

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("LinearOperatorPermutation")
def _np_linearoperatorpermutation(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorPermutation.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorPermutation

    return LinearOperatorPermutation(*args, **kwargs)


@numpy_eager_registry.register("SobolSample")
def _np_sobolsample(backend_module: BackendModuleProtocol, dim: int, num_results: int, skip: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SobolSample eagerly.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        dim (int): Input `dim` argument (dim).
        num_results (int): Input `num_results` argument (num results).
        skip (int): Input `skip` argument (skip).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy as np

    np.random.seed((42 + skip) % (2**32 - 1))
    return backend_module.array(np.random.uniform(size=(num_results, dim)), dtype=backend_module.float32)


@numpy_eager_registry.register("RandomCategorical")
def _np_randomcategorical(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomcategorical operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    logits = backend_module.asarray(args[0])
    num_samples = args[1] if len(args) > 1 else kwargs.get("num_samples", 1)
    return backend_module.zeros(list(logits.shape[:-1]) + [num_samples], dtype=np.int64)


@numpy_eager_registry.register("RandomPermutation")
def _np_randompermutation(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randompermutation operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy as np

    x = backend_module.asarray(args[0])
    if x.ndim == 0:
        return backend_module.array(np.random.permutation(int(x)))
    return backend_module.array(np.random.permutation(np.asarray(x)))


@numpy_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randomtruncatednormal operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return backend_module.random.standard_normal(size=shape)


@numpy_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np randombernoulli operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return backend_module.random.binomial(1, p, size=shape)
