"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Copysign")
def _copysign(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager copysign operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("copysign", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    y = args[1]
    return adapt_backend_module(backend_module).require_symbol("abs")(x) * adapt_backend_module(
        backend_module
    ).require_symbol("sign")(y)


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager getprintoptions operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("get_printoptions")(*args, **kwargs)


@global_eager_registry.register("NpTensorarrayread")
def _np_tensorarrayread(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorarrayread operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional arguments (handle/array, index).
        **kwargs: Keyword arguments.

    Returns:
        BackendArg: Element extracted at the specified index.

    Raises:
        IndexError: If index is out of bounds for the array or list.
    """
    if not args or args[0] is None:
        return 0

    arr = args[0]
    index = int(coerce_int(args[1])) if len(args) > 1 else int(coerce_int(kwargs.get("index", 0)))

    if isinstance(arr, list):
        if index < 0 or index >= len(arr):
            raise IndexError(f"TensorArray index {index} out of bounds for list of length {len(arr)}")
        res_item: BackendArg = arr[index]
        return res_item

    from ml_switcheroo_compiler.backends.eager_registry import BackendArrayProtocol

    if isinstance(arr, BackendArrayProtocol):
        if index < 0 or index >= arr.shape[0]:
            raise IndexError(f"TensorArray index {index} out of bounds for axis 0 of size {arr.shape[0]}")
        return arr[index]

    if isinstance(arr, dict):
        if index not in arr:
            raise IndexError(f"TensorArray index {index} not found in sparse handle")
        res_dict_item: BackendArg = arr[index]
        return res_dict_item

    return 0


@global_eager_registry.register("NpTensorarraywrite")
def _np_tensorarraywrite(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorarraywrite operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional arguments (handle/array, index, value).
        **kwargs: Keyword arguments.

    Returns:
        BackendArg: Updated array or list containing the written element.

    Raises:
        IndexError: If index is negative.
    """
    if not args:
        return 0

    arr = args[0]
    index = int(coerce_int(args[1])) if len(args) > 1 else int(coerce_int(kwargs.get("index", 0)))
    value = args[2] if len(args) > 2 else kwargs.get("value", None)

    if index < 0:
        raise IndexError(f"Negative TensorArray index {index} is invalid")

    if isinstance(arr, list) or arr is None:
        res = list(arr) if arr is not None else []
        if index >= len(res):
            res.extend([None] * (index - len(res) + 1))
        res[index] = value
        return res

    from ml_switcheroo_compiler.backends.eager_registry import BackendArrayProtocol

    mod = _resolve_eager_mod(backend_module)
    raw_nd_cls: BackendArg = mod.ndarray
    if isinstance(raw_nd_cls, type) and isinstance(arr, raw_nd_cls) and isinstance(arr, BackendArrayProtocol):
        if index < arr.shape[0]:
            res_arr = adapt_backend_module(arr).require_op("copy")()
            res_arr[index] = value
            return res_arr
        pad_shape = list(arr.shape)
        pad_shape[0] = index - arr.shape[0] + 1
        pad = mod.zeros(pad_shape, dtype=arr.dtype)
        res_arr = mod.concatenate([arr, pad], axis=0)
        res_arr[index] = value
        return res_arr

    if isinstance(arr, dict):
        res_dict = dict(arr)
        res_dict[index] = value
        return res_dict

    return 0


@global_eager_registry.register("NpTopk")
def _np_topk(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np topk operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr = args[0]
    k = int(coerce_int(args[1])) if len(args) > 1 else 1
    mod = _resolve_eager_mod(backend_module)
    raw_nd_cls_topk: BackendArg = mod.ndarray
    if isinstance(arr, list) or (isinstance(raw_nd_cls_topk, type) and isinstance(arr, raw_nd_cls_topk)):
        arr_np = mod.array(arr)
        idx = mod.argsort(arr_np)[-k:][::-1]
        return arr_np[idx], idx
    return [0], [0]
