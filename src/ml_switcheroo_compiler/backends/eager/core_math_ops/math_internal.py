# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_internal module."""

from __future__ import annotations

import builtins
import typing
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Copysign")
def _copysign(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _copysign operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "copysign", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    y = args[1]
    return backend_module.abs(x) * backend_module.sign(y)


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _getprintoptions operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@global_eager_registry.register("NpTensorarrayread")
def _np_tensorarrayread(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarrayread operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional arguments (handle/array, index).
        **kwargs: Keyword arguments.

    Returns:
        Any: Element extracted at the specified index.

    Raises:
        IndexError: If index is out of bounds for the array or list.
    """
    if not args or args[0] is None:
        return 0

    arr = args[0]
    index = int(args[1]) if len(args) > 1 else int(kwargs.get("index", 0))

    if isinstance(arr, list):
        if index < 0 or index >= len(arr):
            raise IndexError(f"TensorArray index {index} out of bounds for list of length {len(arr)}")
        return arr[index]

    if hasattr(arr, "shape") and hasattr(arr, "__getitem__"):
        if index < 0 or index >= arr.shape[0]:
            raise IndexError(f"TensorArray index {index} out of bounds for axis 0 of size {arr.shape[0]}")
        return arr[index]

    if isinstance(arr, dict):
        if index not in arr:
            raise IndexError(f"TensorArray index {index} not found in sparse handle")
        return arr[index]

    return 0


@global_eager_registry.register("NpTensorarraywrite")
def _np_tensorarraywrite(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarraywrite operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional arguments (handle/array, index, value).
        **kwargs: Keyword arguments.

    Returns:
        Any: Updated array or list containing the written element.

    Raises:
        IndexError: If index is negative.
    """
    if not args:
        return 0

    arr = args[0]
    index = int(args[1]) if len(args) > 1 else int(kwargs.get("index", 0))
    value = args[2] if len(args) > 2 else kwargs.get("value", None)

    if index < 0:
        raise IndexError(f"Negative TensorArray index {index} is invalid")

    if isinstance(arr, list) or arr is None:
        res = list(arr) if arr is not None else []
        if index >= len(res):
            res.extend([None] * (index - len(res) + 1))
        res[index] = value
        return res

    import numpy as np

    if isinstance(arr, np.ndarray):
        if index < arr.shape[0]:
            res_arr = arr.copy()
            res_arr[index] = value
            return res_arr
        pad_shape = list(arr.shape)
        pad_shape[0] = index - arr.shape[0] + 1
        pad = np.zeros(pad_shape, dtype=arr.dtype)
        res_arr = np.concatenate([arr, pad], axis=0)
        res_arr[index] = value
        return res_arr

    if isinstance(arr, dict):
        res_dict = dict(arr)
        res_dict[index] = value
        return res_dict

    return 0


@global_eager_registry.register("NpTopk")
def _np_topk(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_topk operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    arr = args[0]
    k = args[1] if len(args) > 1 else 1
    import numpy as np

    if isinstance(arr, (list, np.ndarray)):
        arr_np = np.array(arr)
        idx = np.argsort(arr_np)[-k:][::-1]
        return arr_np[idx], idx
    return [0], [0]
