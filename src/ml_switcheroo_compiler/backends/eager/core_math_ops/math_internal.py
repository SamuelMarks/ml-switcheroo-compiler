"""math_internal module."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import host_backend
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)


@global_eager_registry.register("Copysign")
def _copysign(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager copysign operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "copysign", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    y = args[1]
    return cast(EagerValue, getattr(backend_module, "abs")(x) * getattr(backend_module, "sign")(y))


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager getprintoptions operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "get_printoptions")(*args, **kwargs))


@global_eager_registry.register("NpTensorarrayread")
def _np_tensorarrayread(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorarrayread operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional arguments (handle/array, index).
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Element extracted at the specified index.

    Raises:
        IndexError: If index is out of bounds for the array or list.
    """
    if not args or args[0] is None:
        return 0

    arr = args[0]
    index = int(cast(int, args[1])) if len(args) > 1 else int(cast(int, kwargs.get("index", 0)))

    if isinstance(arr, list):
        if index < 0 or index >= len(arr):
            raise IndexError(f"TensorArray index {index} out of bounds for list of length {len(arr)}")
        return cast(EagerValue, arr[index])

    if hasattr(arr, "shape") and hasattr(arr, "__getitem__"):
        arr_t = typing.cast(EagerTensorProtocol, arr)
        if index < 0 or index >= arr_t.shape[0]:
            raise IndexError(f"TensorArray index {index} out of bounds for axis 0 of size {arr_t.shape[0]}")
        return arr_t[index]

    if isinstance(arr, dict):
        if index not in arr:
            raise IndexError(f"TensorArray index {index} not found in sparse handle")
        return cast(EagerValue, arr[index])

    return 0


@global_eager_registry.register("NpTensorarraywrite")
def _np_tensorarraywrite(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorarraywrite operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional arguments (handle/array, index, value).
        **kwargs: Keyword arguments.

    Returns:
        EagerValue: Updated array or list containing the written element.

    Raises:
        IndexError: If index is negative.
    """
    if not args:
        return 0

    arr = args[0]
    index = int(cast(int, args[1])) if len(args) > 1 else int(cast(int, kwargs.get("index", 0)))
    value = args[2] if len(args) > 2 else kwargs.get("value", None)

    if index < 0:
        raise IndexError(f"Negative TensorArray index {index} is invalid")

    if isinstance(arr, list) or arr is None:
        res = list(arr) if arr is not None else []
        if index >= len(res):
            res.extend([None] * (index - len(res) + 1))
        res[index] = value
        return res

    if isinstance(arr, host_backend.ndarray):
        if index < arr.shape[0]:
            res_arr = arr.copy()
            res_arr[index] = value
            return cast(EagerValue, res_arr)
        pad_shape = list(arr.shape)
        pad_shape[0] = index - arr.shape[0] + 1
        pad = host_backend.zeros(pad_shape, dtype=arr.dtype)
        res_arr = host_backend.concatenate([arr, pad], axis=0)
        res_arr[index] = value
        return cast(EagerValue, res_arr)

    if isinstance(arr, dict):
        res_dict = dict(arr)
        res_dict[index] = value
        return res_dict

    return 0


@global_eager_registry.register("NpTopk")
def _np_topk(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np topk operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    arr = args[0]
    k = int(cast(int, args[1])) if len(args) > 1 else 1
    if isinstance(arr, (list, host_backend.ndarray)):
        arr_np = host_backend.array(arr)
        idx = host_backend.argsort(arr_np)[-k:][::-1]
        return arr_np[idx], idx
    return [0], [0]
