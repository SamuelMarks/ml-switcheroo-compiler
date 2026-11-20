"""Generic backend-parametric eager reference operations for `reference_math.math_special_generic`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("Bartlett")
def _np_bartlett(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    M: int,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Return the Bartlett window.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        M (int): Number of points in the output window.
        **kwargs (Union[float, int, str, bool, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: The triangular Bartlett window.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(adapt_backend_module(arr_mod).require_symbol("bartlett")(M))
