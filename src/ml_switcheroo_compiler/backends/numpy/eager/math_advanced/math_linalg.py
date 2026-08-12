# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_linalg module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Rsqrt")
def _np_rsqrt(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rsqrt operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(x)


@numpy_eager_registry.register("Trace")
def _np_trace(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the sum along diagonals of the array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.trace(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Vander")
def _np_vander(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vander operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.vander(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Cholesky")
def _np_linalg_cholesky_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Cholesky via linalg.cholesky.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.cholesky(*args, **kwargs)


@numpy_eager_registry.register("Det")
def _np_linalg_det_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Det via linalg.det.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.det(*args, **kwargs)


@numpy_eager_registry.register("Svd")
def _np_linalg_svd_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Svd via linalg.svd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.svd(*args, **kwargs)


@numpy_eager_registry.register("Inv")
def _np_linalg_inv_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Inv via linalg.inv.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.inv(*args, **kwargs)


@numpy_eager_registry.register("Pinv")
def _np_linalg_pinv_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Pinv via linalg.pinv.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.pinv(*args, **kwargs)


@numpy_eager_registry.register("CustomLinearSolve")
def _np_customlinearsolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement CustomLinearSolve.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if callable(args[0]):
        solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
        if solve:
            return solve(args[0], args[1])
        return args[1]
    return np.linalg.solve(args[0], args[1])


@numpy_eager_registry.register("LinearOperatorInversion")
def _np_linearoperatorinversion(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorInversion.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorInversion

    return LinearOperatorInversion(*args, **kwargs)


@numpy_eager_registry.register("Vecdot")
def _np_vecdot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Vecdot.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
    return backend_module.sum(x * y, axis=axis)
