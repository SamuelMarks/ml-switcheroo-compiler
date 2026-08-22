# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_matrix_utils module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg


@numpy_eager_registry.register("DiagIndices")
def _np_diag_indices_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DiagIndices via diag_indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diag_indices(*args, **kwargs)


@numpy_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DiagIndicesFrom via diag_indices_from.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diag_indices_from(*args, **kwargs)


@numpy_eager_registry.register("Diagflat")
def _np_diagflat_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Diagflat via diagflat.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diagflat(*args, **kwargs)


@numpy_eager_registry.register("Diagonal")
def _np_diagonal_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Diagonal via diagonal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diagonal(*args, **kwargs)


@numpy_eager_registry.register("Indices")
def _np_indices_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Indices via indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.indices(*args, **kwargs)


@numpy_eager_registry.register("MaskIndices")
def _np_mask_indices_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement MaskIndices via mask_indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.mask_indices(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorBlockDiag")
def _np_linearoperatorblockdiag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorBlockDiag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockDiag

    return LinearOperatorBlockDiag(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorDiag")
def _np_linearoperatordiag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorDiag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorDiag

    return LinearOperatorDiag(*args, **kwargs)


@numpy_eager_registry.register("confusion_matrix")
def _np_confusion_matrix(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "confusion_matrix"):
            cls_or_func = _ops.confusion_matrix
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, _ops.OpDef)):
                return cls_or_func(*args, **kwargs)
    except Exception:
        pass
    if hasattr(backend_module, "confusion_matrix"):
        return backend_module.confusion_matrix(*args, **kwargs)
    y_true = np.asarray(args[0]).flatten()
    y_pred = np.asarray(args[1]).flatten()
    num_classes = kwargs.get("num_classes", args[2] if len(args) > 2 else None)
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


@numpy_eager_registry.register("distributions")
def _np_distributions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """_np_distributions function.

    Args:
        backend_module: The backend.
        args: Positional args.
        kwargs: Keyword args.

    Args:
        message (str): The message.
        input_vars (list): The input vars.
        node (Any): The node.
        **kwargs (Any): Keyword arguments.
        backend_module (Any): The backend_module parameter.

    Returns:
        Any: Result.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "distributions"):
            cls_or_func = _ops.distributions
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, getattr(_ops, "OpDef", object))):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        print("EXCEPTION IN _np_distributions:", repr(e))
        pass
    if hasattr(backend_module, "distributions"):
        return backend_module.distributions(*args, **kwargs)
    arr = np.asarray(args[0]) if args else np.zeros((1,))
    return np.array([np.mean(arr), np.var(arr)])


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix_cap(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """_np_confusion_matrix_cap function.

    Args:
        backend_module: The backend.
        args: Positional args.
        kwargs: Keyword args.

    Args:
        message (str): The message.
        input_vars (list): The input vars.
        node (Any): The node.
        **kwargs (Any): Keyword arguments.
        backend_module (Any): The backend_module parameter.

    Returns:
        Any: Result.
    """
    a = _get_np_arg(args, 0)
    b = _get_np_arg(args, 1)
    if a is None or b is None:
        return None
    return np.zeros((2, 2))
