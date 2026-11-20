"""Generic backend-parametric eager reference operations for `reference_math.math_set`."""

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


@global_eager_registry.register("Union1d")
def _np_union1d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: bool | None,
) -> EagerTensorProtocol:
    """Find the union of two one-dimensional arrays.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: The computed unique union.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.union1d(arr_mod.asarray(args[0]), arr_mod.asarray(args[1]), **kwargs)


@global_eager_registry.register("Intersect1d")
def _np_intersect1d_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: bool | None,
) -> EagerTensorProtocol:
    """Implement Intersect1d via intersect1d.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: The computed intersection of two arrays.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("intersect1d")(*args, **kwargs))


@global_eager_registry.register("Isin")
def _np_isin_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: bool | None,
) -> EagerTensorProtocol:
    """Implement Isin via isin.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[bool, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: Boolean array of element containment.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("isin")(*args, **kwargs))
