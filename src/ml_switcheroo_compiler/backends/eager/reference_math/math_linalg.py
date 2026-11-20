"""Generic backend-parametric eager reference operations for `reference_math.math_linalg`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Rsqrt")
def _np_rsqrt(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np rsqrt operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    with adapt_backend_module(arr_mod).require_symbol("errstate")(divide="ignore", invalid="ignore"):
        return 1.0 / adapt_backend_module(arr_mod).require_symbol("sqrt")(x)


@global_eager_registry.register("Trace")
def _np_trace(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Return the sum along diagonals of the array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("trace")(arr_mod.asarray(args[0]), *args[1:], **kwargs)


@global_eager_registry.register("Vander")
def _np_vander(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np vander operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("vander")(arr_mod.asarray(args[0]), *args[1:], **kwargs)


@global_eager_registry.register("Cholesky")
def _np_linalg_cholesky_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Cholesky via linalg.cholesky.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("linalg").cholesky(*args, **kwargs)


@global_eager_registry.register("Det")
def _np_linalg_det_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Det via linalg.det.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("linalg").det(*args, **kwargs)


@global_eager_registry.register("Svd")
def _np_linalg_svd_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Svd via linalg.svd.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("linalg").svd(*args, **kwargs)


@global_eager_registry.register("Inv")
def _np_linalg_inv_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Inv via linalg.inv.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("linalg").inv(*args, **kwargs)


@global_eager_registry.register("Pinv")
def _np_linalg_pinv_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Pinv via linalg.pinv.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("linalg").pinv(*args, **kwargs)


@global_eager_registry.register("CustomLinearSolve")
def _np_customlinearsolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement CustomLinearSolve.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if callable(args[0]):
        solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
        if solve:
            return require_callable(solve)(args[0], args[1])
        return args[1]
    return adapt_backend_module(arr_mod.linalg).require_symbol("solve")(args[0], args[1])


@global_eager_registry.register("LinearOperatorInversion")
def _np_linearoperatorinversion(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorInversion.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorInversion

    return LinearOperatorInversion(*args, **kwargs)


@global_eager_registry.register("Vecdot")
def _np_vecdot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Vecdot.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    x = args[0]
    y = args[1]
    axis = kwargs.get("axis", -1)
    if adapt_backend_module(backend_module).has_symbol("linalg") and adapt_backend_module(
        adapt_backend_module(backend_module).require_symbol("linalg")
    ).has_symbol("vecdot"):
        _typed_ret_1: BackendArg = adapt_backend_module(backend_module).require_symbol("linalg").vecdot(x, y, axis=axis)
        return _typed_ret_1
    if adapt_backend_module(backend_module).has_symbol("vecdot"):
        _typed_ret_2: BackendArg = adapt_backend_module(backend_module).require_symbol("vecdot")(x, y, axis=axis)
        return _typed_ret_2
    if adapt_backend_module(backend_module).require_symbol("iscomplexobj")(x):
        x = adapt_backend_module(backend_module).require_symbol("conj")(x)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    return adapt_backend_module(backend_module).require_symbol("sum")(coerce_numeric(x) * coerce_numeric(y), axis=axis)
