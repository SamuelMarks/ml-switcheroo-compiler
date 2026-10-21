"""Implementation of math linalg operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_linalg`)."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: F401

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_tensor, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("Rsqrt")
def _np_rsqrt(backend_module: BackendModuleProtocol, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rsqrt operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return as_tensor(1.0 / getattr(np, "sqrt")(x))


@numpy_eager_registry.register("Trace")
def _np_trace(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Return the sum along diagonals of the array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return as_tensor(getattr(np, "trace")(np.asarray(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("Vander")
def _np_vander(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vander operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return as_tensor(getattr(np, "vander")(np.asarray(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("Cholesky")
def _np_linalg_cholesky_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Cholesky via linalg.cholesky.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.linalg.cholesky(*args, **kwargs)


@numpy_eager_registry.register("Det")
def _np_linalg_det_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Det via linalg.det.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.linalg.det(*args, **kwargs)


@numpy_eager_registry.register("Svd")
def _np_linalg_svd_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Svd via linalg.svd.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.linalg.svd(*args, **kwargs)


@numpy_eager_registry.register("Inv")
def _np_linalg_inv_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Inv via linalg.inv.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.linalg.inv(*args, **kwargs)


@numpy_eager_registry.register("Pinv")
def _np_linalg_pinv_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Pinv via linalg.pinv.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.linalg.pinv(*args, **kwargs)


@numpy_eager_registry.register("CustomLinearSolve")
def _np_customlinearsolve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement CustomLinearSolve.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if callable(args[0]):
        solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
        if solve:
            return as_tensor(as_callable(solve)(args[0], args[1]))
        return args[1]
    return as_tensor(getattr(np.linalg, "solve")(args[0], args[1]))


@numpy_eager_registry.register("LinearOperatorInversion")
def _np_linearoperatorinversion(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorInversion.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorInversion

    return LinearOperatorInversion(*args, **kwargs)


@numpy_eager_registry.register("Vecdot")
def _np_vecdot(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Vecdot.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x = args[0]
    y = args[1]
    axis = kwargs.get("axis", -1)
    if hasattr(backend_module, "linalg") and hasattr(backend_module.linalg, "vecdot"):
        return backend_module.linalg.vecdot(x, y, axis=axis)
    if hasattr(backend_module, "vecdot"):
        return backend_module.vecdot(x, y, axis=axis)
    if backend_module.iscomplexobj(x):
        x = backend_module.conj(x)
    return as_tensor(backend_module.sum(as_tensor(x) * as_tensor(y), axis=axis))
