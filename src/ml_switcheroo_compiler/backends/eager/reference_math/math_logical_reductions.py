"""Generic backend-parametric eager reference operations for `reference_math.math_logical_reductions`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
    adapt_backend_module,
)


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("All")
def _np_all(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np all operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to all.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Evaluated boolean reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("all")(*args, **kwargs))


@global_eager_registry.register("CountNonzero")
def _np_count_nonzero(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np count nonzero operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to count_nonzero.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Count of non-zero elements.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("count_nonzero")(*args, **kwargs))
