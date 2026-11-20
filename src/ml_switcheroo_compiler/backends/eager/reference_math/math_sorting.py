"""Generic backend-parametric eager reference operations for `reference_math.math_sorting`."""

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


@global_eager_registry.register("ArgPartition")
def _np_argpartition(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | tuple[int, ...] | str | None,
) -> EagerTensorProtocol:
    """Perform an indirect partition along the given axis.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[int, tuple[int, ...], str, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: The computed partition indices.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("argpartition")(*args, **kwargs))


@global_eager_registry.register("Lexsort")
def _np_lexsort_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> EagerTensorProtocol:
    """Implement Lexsort via lexsort.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[int, str, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: The computed lexicographical sort order.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("lexsort")(*args, **kwargs))


@global_eager_registry.register("Median")
def _np_median_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Implement Median via median.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Variable positional arguments.
        **kwargs (Union[int, tuple[int, ...], bool, None]): Arbitrary keyword arguments.

    Returns:
        EagerTensorProtocol: The computed median value.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("median")(*args, **kwargs))
