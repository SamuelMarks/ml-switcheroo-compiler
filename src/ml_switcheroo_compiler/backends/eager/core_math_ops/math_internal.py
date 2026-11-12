"""Implementation of math internal operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    ensure_eager_tensor,
    ensure_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member, has_backend_member


def _as_tensor_cls(cls_val: EagerValue) -> type[EagerTensorProtocol]:
    """Narrow `cls_val` to `type[EagerTensorProtocol]`.

    Args:
        cls_val (EagerValue): Candidate array class.

    Returns:
        type[EagerTensorProtocol]: Typed array class.
    """

    def _guard(cand: EagerValue) -> typing.TypeGuard[type[EagerTensorProtocol]]:
        """Type guard narrowing `cand` to `type[EagerTensorProtocol]`.

        Args:
            cand (EagerValue): Candidate class object.

        Returns:
            bool: True for runtime ndarray classes.
        """
        return not isinstance(cand, type(Ellipsis))

    assert _guard(cls_val)
    return cls_val


@global_eager_registry.register("Copysign")
def _copysign(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager copysign operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "copysign", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))
    x = args[0]
    y = args[1]
    return ensure_eager_tensor(get_backend_member(backend_module, "abs")(x) * get_backend_member(backend_module, "sign")(y))


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager getprintoptions operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "get_printoptions")(*args, **kwargs))


@global_eager_registry.register("NpTensorarrayread")
def _np_tensorarrayread(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorarrayread operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
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
    index = int(ensure_int(args[1])) if len(args) > 1 else int(ensure_int(kwargs.get("index", 0)))

    if isinstance(arr, list):
        if index < 0 or index >= len(arr):
            raise IndexError(f"TensorArray index {index} out of bounds for list of length {len(arr)}")
        return ensure_eager_tensor(arr[index])

    if has_backend_member(arr, "shape"):
        arr_t = ensure_eager_tensor(arr)
        if index < 0 or index >= arr_t.shape[0]:
            raise IndexError(f"TensorArray index {index} out of bounds for axis 0 of size {arr_t.shape[0]}")
        return arr_t[index]

    if isinstance(arr, dict):
        if index not in arr:
            raise IndexError(f"TensorArray index {index} not found in sparse handle")
        return ensure_eager_tensor(arr[index])

    return 0


@global_eager_registry.register("NpTensorarraywrite")
def _np_tensorarraywrite(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorarraywrite operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
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
    index = int(ensure_int(args[1])) if len(args) > 1 else int(ensure_int(kwargs.get("index", 0)))
    value = args[2] if len(args) > 2 else kwargs.get("value", None)

    if index < 0:
        raise IndexError(f"Negative TensorArray index {index} is invalid")

    if isinstance(arr, list) or arr is None:
        res = list(arr) if arr is not None else []
        if index >= len(res):
            res.extend([None] * (index - len(res) + 1))
        res[index] = value
        return res

    mod = _resolve_eager_mod(backend_module)
    ndarray_cls = _as_tensor_cls(mod.ndarray)
    if isinstance(arr, ndarray_cls):
        if index < arr.shape[0]:
            res_arr = arr.copy()
            res_arr[index] = value
            _typed_ret_1: EagerValue = res_arr
            return _typed_ret_1
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
def _np_topk(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np topk operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    arr = args[0]
    k = int(ensure_int(args[1])) if len(args) > 1 else 1
    mod = _resolve_eager_mod(backend_module)
    ndarray_cls = _as_tensor_cls(mod.ndarray)
    if isinstance(arr, (list, ndarray_cls)):
        arr_np = mod.array(arr)
        idx = mod.argsort(arr_np)[-k:][::-1]
        return arr_np[idx], idx
    return [0], [0]
