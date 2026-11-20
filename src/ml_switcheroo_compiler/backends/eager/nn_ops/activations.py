"""Generic backend-parametric eager reference operations for `nn_ops.activations`."""

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
        BackendArg: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("Relu")
def _np_relu(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: float | int | str | bool | None,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np relu operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        *args (Union[float, int, str, bool, None]): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("maximum")(x, 0.0))


@global_eager_registry.register("Elu")
def _np_elu(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    alpha: float = 1.0,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np elu operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        alpha (float): Input `alpha` argument (alpha).
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x_arr = coerce_numeric(x)
    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol("where")(
            x_arr > 0, x_arr, alpha * (adapt_backend_module(backend_module).require_symbol("exp")(x_arr) - 1.0)
        )
    )


@global_eager_registry.register("Celu")
def _np_celu(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    alpha: float = 1.0,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np celu operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        alpha (float): Input `alpha` argument (alpha).
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x_arr = coerce_numeric(x)
    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol("maximum")(0.0, x_arr)
        + adapt_backend_module(backend_module).require_symbol("minimum")(
            0.0, alpha * (adapt_backend_module(backend_module).require_symbol("exp")(x_arr / alpha) - 1.0)
        )
    )


@global_eager_registry.register("Softplus")
def _np_softplus(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np softplus operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol("log1p")(
            adapt_backend_module(backend_module).require_symbol("exp")(
                -adapt_backend_module(backend_module).require_symbol("abs")(x)
            )
        )
        + adapt_backend_module(backend_module).require_symbol("maximum")(x, 0.0)
    )


@global_eager_registry.register("Softsign")
def _np_softsign(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np softsign operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x_arr = coerce_numeric(x)
    return _ensure_arr(x_arr / (1.0 + adapt_backend_module(backend_module).require_symbol("abs")(x_arr)))


@global_eager_registry.register("Mish")
def _np_mish(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np mish operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x_arr = coerce_numeric(x)
    softplus_x = adapt_backend_module(backend_module).require_symbol("log1p")(
        adapt_backend_module(backend_module).require_symbol("exp")(
            -adapt_backend_module(backend_module).require_symbol("abs")(x_arr)
        )
    ) + adapt_backend_module(backend_module).require_symbol("maximum")(x_arr, 0.0)
    return _ensure_arr(x_arr * adapt_backend_module(backend_module).require_symbol("tanh")(softplus_x))


@global_eager_registry.register("LogSigmoid")
def _np_log_sigmoid(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np log sigmoid operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Computed `EagerTensorProtocol` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x_arr = coerce_numeric(x)
    return _ensure_arr(
        -adapt_backend_module(backend_module).require_symbol("log1p")(
            adapt_backend_module(backend_module).require_symbol("exp")(-x_arr)
        )
    )
