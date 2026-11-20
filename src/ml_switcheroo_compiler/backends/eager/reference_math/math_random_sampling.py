"""Generic backend-parametric eager reference operations for `reference_math.math_random_sampling`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("LinearOperatorPermutation")
def _np_linearoperatorpermutation(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorPermutation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorPermutation

    return LinearOperatorPermutation(*args, **kwargs)


@global_eager_registry.register("SobolSample")
def _np_sobolsample(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    dim: int,
    num_results: int,
    skip: int = 0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Implement SobolSample eagerly.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        dim (int): Input `dim` argument (dim).
        num_results (int): Input `num_results` argument (num results).
        skip (int): Input `skip` argument (skip).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    adapt_backend_module(arr_mod.random).require_symbol("seed")((42 + skip) % (2**32 - 1))
    return adapt_backend_module(backend_module).require_symbol("array")(
        arr_mod.random.uniform(size=(num_results, dim)),
        dtype=adapt_backend_module(backend_module).require_symbol("float32"),
    )


@global_eager_registry.register("RandomCategorical")
def _np_randomcategorical(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomcategorical operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    logits = adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    num_samples = args[1] if len(args) > 1 else kwargs.get("num_samples", 1)
    return adapt_backend_module(backend_module).require_symbol("zeros")(
        list(logits.shape[:-1]) + [num_samples], dtype=arr_mod.int64
    )


@global_eager_registry.register("RandomPermutation")
def _np_randompermutation(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randompermutation operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    if x.ndim == 0:
        return adapt_backend_module(backend_module).require_symbol("array")(
            adapt_backend_module(arr_mod.random).require_symbol("permutation")(int(x))
        )
    return adapt_backend_module(backend_module).require_symbol("array")(
        adapt_backend_module(arr_mod.random).require_symbol("permutation")(arr_mod.asarray(x))
    )


@global_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randomtruncatednormal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return adapt_backend_module(backend_module).require_symbol("random").standard_normal(size=shape)


@global_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np randombernoulli operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return adapt_backend_module(backend_module).require_symbol("random").binomial(1, p, size=shape)
