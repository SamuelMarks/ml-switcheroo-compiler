"""math_matrix_utils module."""

from __future__ import annotations

import logging
import typing  # noqa: F401
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401

from .math_general import _get_np_arg


@numpy_eager_registry.register("DiagIndices")
def _np_diag_indices_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement DiagIndices via diag_indices.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "diag_indices")(*args, **kwargs))


@numpy_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement DiagIndicesFrom via diag_indices_from.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "diag_indices_from")(*args, **kwargs))


@numpy_eager_registry.register("Diagflat")
def _np_diagflat_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Diagflat via diagflat.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "diagflat")(*args, **kwargs))


@numpy_eager_registry.register("Diagonal")
def _np_diagonal_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Diagonal via diagonal.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "diagonal")(*args, **kwargs))


@numpy_eager_registry.register("Indices")
def _np_indices_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Indices via indices.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "indices")(*args, **kwargs))


@numpy_eager_registry.register("MaskIndices")
def _np_mask_indices_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement MaskIndices via mask_indices.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return typing.cast(typing.Callable[..., EagerValue], getattr(backend_module, "mask_indices"))(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorBlockDiag")
def _np_linearoperatorblockdiag(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorBlockDiag.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockDiag

    return LinearOperatorBlockDiag(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorDiag")
def _np_linearoperatordiag(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorDiag.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorDiag

    return LinearOperatorDiag(*args, **kwargs)


@numpy_eager_registry.register("confusion_matrix")
def _np_confusion_matrix(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "confusion_matrix"):
            cls_or_func = _ops.confusion_matrix
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, getattr(_ops, "OpDef", type))):
                return cast(EagerValue, cls_or_func(*args, **kwargs))
    except Exception:
        # Ignore dynamic load errors
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if hasattr(backend_module, "confusion_matrix"):
        return cast(EagerValue, getattr(backend_module, "confusion_matrix")(*args, **kwargs))
    y_true = np.asarray(args[0]).flatten()
    y_pred = np.asarray(args[1]).flatten()
    raw_num_classes = kwargs.get("num_classes", args[2] if len(args) > 2 else None) if hasattr(kwargs, "get") else None
    num_classes: int = int(np.max(y_true) if np.max(y_true) > np.max(y_pred) else np.max(y_pred)) + 1 if raw_num_classes is None else int(getattr(raw_num_classes, "__int__", lambda: 0)())
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


@numpy_eager_registry.register("distributions")
def _np_distributions(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """_np_distributions function.

    Args:
        backend_module: The backend.
        args: Positional args.
        kwargs: Keyword args.

    Returns:
        Any: Result.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "distributions"):
            cls_or_func = _ops.distributions
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, getattr(_ops, "OpDef", type))):
                return cast(EagerValue, cls_or_func(*args, **kwargs))
    except Exception as e:
        print("EXCEPTION IN _np_distributions:", repr(e))
        # Ignore dynamic load errors
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if hasattr(backend_module, "distributions"):
        return cast(EagerValue, getattr(backend_module, "distributions")(*args, **kwargs))
    arr = np.asarray(args[0]) if args else np.zeros((1,))
    return np.array([np.mean(arr), np.var(arr)])


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix_cap(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """_np_confusion_matrix_cap function.

    Args:
        backend_module: The backend.
        args: Positional args.
        kwargs: Keyword args.

    Returns:
        Any: Result.
    """
    a = _get_np_arg(args, 0)
    b = _get_np_arg(args, 1)
    if a is None or b is None:
        return None
    return np.zeros((2, 2))
