"""Generic backend-parametric eager reference operations for `reference_math.math_log_exp`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg  # noqa: F401
from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Xlogy")
def _np_xlogy(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np xlogy operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _xlogy(*args, **kwargs)


@global_eager_registry.register("Logsumexp")
def _np_logsumexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    a: BackendArg,
    axis: BackendArg = None,
    keepdims: bool = False,
    **kwargs: BackendArg,
) -> BackendArg:
    """Evaluate _np_logsumexp logic eagerly backed by NumPy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        a (BackendArg): Input `a` argument (a).
        axis (BackendArg): Input `axis` argument (axis).
        keepdims (bool): Input `keepdims` argument (keepdims).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = arr_mod.array(a)
    a_max = adapt_backend_module(arr_mod).require_symbol("amax")(a, axis=axis, keepdims=True)
    if not keepdims:
        a_max_s = adapt_backend_module(arr_mod).require_symbol("squeeze")(a_max, axis=axis)
    else:
        a_max_s = a_max
    out = arr_mod.log(
        adapt_backend_module(arr_mod).require_symbol("sum")(arr_mod.exp(a - a_max), axis=axis, keepdims=keepdims)
    )
    out += a_max_s
    return out


@global_eager_registry.register("Log1P")
def _np_log1p2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np log1p2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("log1p")(*args, **kwargs)


@global_eager_registry.register("Unsqueeze")
def _np_expand_dims_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Unsqueeze via expand_dims.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("expand_dims")(*args, **kwargs)


@global_eager_registry.register("LogSoftmax")
def _np_log_softmax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np log softmax operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    c = adapt_backend_module(arr_mod).require_symbol("max")(a, axis=axis, keepdims=True)
    return (
        a
        - c
        - arr_mod.log(adapt_backend_module(arr_mod).require_symbol("sum")(arr_mod.exp(a - c), axis=axis, keepdims=True))
    )
