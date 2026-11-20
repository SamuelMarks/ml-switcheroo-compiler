"""Generic backend-parametric eager reference operations for `reference_math.math_nan`."""

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


@global_eager_registry.register("DivideNoNan")
def _np_divide_no_nan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, y: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """DivideNoNan.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("divide")(
        x,
        y,
        out=adapt_backend_module(backend_module).require_symbol("zeros_like")(
            x, dtype=adapt_backend_module(backend_module).require_symbol("result_type")(x, y)
        ),
        where=y != 0,
    )


@global_eager_registry.register("MultiplyNoNan")
def _np_multiply_no_nan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, y: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """MultiplyNoNan.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("multiply")(
        x,
        y,
        out=adapt_backend_module(backend_module).require_symbol("zeros_like")(
            x, dtype=adapt_backend_module(backend_module).require_symbol("result_type")(x, y)
        ),
        where=y != 0,
    )


@global_eager_registry.register("SquaredDifference")
def _np_squared_difference(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, y: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """SquaredDifference.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    diff = adapt_backend_module(backend_module).require_symbol("subtract")(x, y)
    return adapt_backend_module(backend_module).require_symbol("square")(diff)


@global_eager_registry.register("Xdivy")
def _np_xdivy(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, y: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Xdivy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("divide")(
        x,
        y,
        out=adapt_backend_module(backend_module).require_symbol("zeros_like")(
            x, dtype=adapt_backend_module(backend_module).require_symbol("result_type")(x, y)
        ),
        where=x != 0,
    )


@global_eager_registry.register("Xlog1py")
def _np_xlog1py(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, y: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Xlog1py.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("multiply")(
        x,
        adapt_backend_module(backend_module).require_symbol("log1p")(y),
        out=adapt_backend_module(backend_module).require_symbol("zeros_like")(
            x, dtype=adapt_backend_module(backend_module).require_symbol("result_type")(x, y)
        ),
        where=x != 0,
    )


@global_eager_registry.register("ReciprocalNoNan")
def _np_reciprocal_no_nan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """ReciprocalNoNan.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("divide")(
        1.0,
        x,
        out=adapt_backend_module(backend_module).require_symbol("zeros_like")(
            x, dtype=adapt_backend_module(backend_module).require_symbol("result_type")(x, 1.0)
        ),
        where=x != 0,
    )


@global_eager_registry.register("ZeroFraction")
def _np_zero_fraction(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """ZeroFraction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    num_zeros = adapt_backend_module(backend_module).require_symbol("sum")(
        adapt_backend_module(backend_module).require_symbol("equal")(x, 0)
    )
    total_elements = adapt_backend_module(backend_module).require_symbol("size")(x)
    if total_elements == 0:
        return adapt_backend_module(backend_module).require_symbol("array")(float("nan"))
    return adapt_backend_module(backend_module).require_symbol("array")(
        adapt_backend_module(backend_module).require_symbol("divide")(num_zeros, total_elements).astype(float)
    )


def _xlogy(x: BackendArg, y: BackendArg) -> EagerTensorProtocol:
    """Execute the eager xlogy operation on the active backend array.

    Args:
        x (BackendArg): Input `x` argument (x).
        y (BackendArg): Input `y` argument (y).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    res = adapt_backend_module(arr_mod).require_symbol("where")(
        x == 0.0, 0.0, x * adapt_backend_module(arr_mod).require_symbol("log")(y)
    )
    return _ensure_arr(res)


@global_eager_registry.register("Nanmean")
def _np_nanmean(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    a: BackendArg,
    axis: BackendArg = None,
    keepdims: BackendArg = False,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Nanmean.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        a (BackendArg): Input `a` argument (a).
        axis (BackendArg): Input `axis` argument (axis).
        keepdims (BackendArg): Input `keepdims` argument (keepdims).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("nanmean")(a, axis=axis, keepdims=keepdims)


@global_eager_registry.register("Nanmedian")
def _np_nanmedian(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    a: BackendArg,
    axis: BackendArg = None,
    keepdims: BackendArg = False,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Nanmedian.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        a (BackendArg): Input `a` argument (a).
        axis (BackendArg): Input `axis` argument (axis).
        keepdims (BackendArg): Input `keepdims` argument (keepdims).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("nanmedian")(a, axis=axis, keepdims=keepdims)
