"""Generic backend-parametric eager reference operations for `control_flow_ops`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
    require_callable,
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


@global_eager_registry.register("AssociativeScan")
def _np_associative_scan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np associative scan operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) < 2:
        return args[0]
    fn = args[0]
    elems: list[BackendArg] = coerce_operand_list(args[1])
    axis: int = coerce_int(kwargs.get("axis", 0))

    elems_arr: EagerTensorProtocol = arr_mod.asarray(elems)
    out: EagerTensorProtocol = arr_mod.empty_like(elems_arr)

    elems_arr = arr_mod.moveaxis(elems_arr, axis, 0)
    out = arr_mod.moveaxis(out, axis, 0)

    acc: EagerTensorProtocol = elems_arr[0]
    out[0] = acc
    for i in range(1, elems_arr.shape[0]):
        acc = require_callable(_ensure_arr)(require_callable(fn)(acc, elems_arr[i]))
        out[i] = acc

    out = arr_mod.moveaxis(out, 0, axis)
    return out


@global_eager_registry.register("StopGradient")
def _stop_gradient(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager stop gradient operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("array")(x)
