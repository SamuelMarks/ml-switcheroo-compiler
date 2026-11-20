"""Generic backend-parametric eager reference operations for `reference_math.math_gamma`."""

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


@global_eager_registry.register("Mvlgamma")
def _np_mvlgamma(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float,
) -> EagerTensorProtocol:
    """Execute the eager np mvlgamma operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Input arguments.
        **kwargs (Union[int, float]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Log multivariate gamma evaluation.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("mvlgamma")(*args, **kwargs))
