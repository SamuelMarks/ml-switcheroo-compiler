"""Eager mode implementations for control flow operations."""

from __future__ import annotations

from typing import Any, Callable

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap


def cond_eager(pred: Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> object:
    """Docstring."""
    if bool(pred.data):
        return true_fn()
    return false_fn()


def while_loop_eager(cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: object) -> object:
    """Docstring."""
    val = init_val
    res = cond_fn(val)
    while bool(res.data if hasattr(res, "data") else res):
        val = body_fn(val)
        res = cond_fn(val)

    return val


def _stack_scan_outputs(ys: list, init: object, last_y: object) -> Tensor:
    """Docstring."""
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(
            stacked_ys,
            TensorConfig(
                stacked_ys.shape,
                last_y.dtype if hasattr(last_y, "dtype") else init.dtype,
                last_y.device if hasattr(last_y, "device") else init.device,
            ),
        )
    else:
        stacked_ys = get_active_backend().array(ys)
        return Tensor(
            stacked_ys,
            TensorConfig(stacked_ys.shape, DType(str(stacked_ys.dtype)), config.default_device),
        )


def scan_eager(f: Callable, init: object, xs: object, length: int | None = None) -> tuple[object, object]:
    """Docstring."""
    carry = init
    ys = []
    scan_length = length if length is not None else (xs.shape[0] if xs is not None else 0)
    for i in range(scan_length):
        x = Tensor(xs.data[i], TensorConfig(xs.shape[1:], xs.dtype, xs.device)) if xs is not None else None
        carry, y = f(carry, x)
        ys.append(y.data if hasattr(y, "data") else y)

    out_tensor = _stack_scan_outputs(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _map_fn_eager_get_length(elems: Tensor) -> int:
    """Function docstring."""
    return elems.shape[0] if elems is not None and len(elems.shape) > 0 else 0


def _map_fn_eager_execute(fn: Callable, elems: Tensor, length: int) -> list[Any]:
    """Function docstring."""
    ys = []
    for i in range(length):
        x = Tensor(elems.data[i], TensorConfig(elems.shape[1:], elems.dtype, elems.device))
        y = fn(x)
        ys.append(y.data if hasattr(y, "data") else y)
    return ys


def _map_fn_eager_stack(ys: list[Any], elems: Tensor, dtype: DType | None) -> Tensor:
    """Function docstring."""
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(stacked_ys, TensorConfig(stacked_ys.shape, elems.dtype, elems.device))

    stacked_ys = get_active_backend().array(ys)
    out_dtype = dtype if dtype is not None else DType(str(stacked_ys.dtype))
    return Tensor(stacked_ys, TensorConfig(stacked_ys.shape, out_dtype, elems.device))


def map_fn_eager(fn: Callable, elems: Tensor, dtype: DType | None = None) -> Tensor:
    """Docstring."""
    length = _map_fn_eager_get_length(elems)
    ys = _map_fn_eager_execute(fn, elems, length)
    return _map_fn_eager_stack(ys, elems, dtype)


def pmap_eager(func: Callable, axis_name: str | None = None) -> Callable:
    """Docstring."""

    def wrapped(*args: object) -> object:
        """Function docstring.

        Args:
        args: Arg.
        """
        return vmap(func)(*args)

    return wrapped


def stop_gradient_eager(x: object) -> object:
    """Docstring."""
    return x


def assert_value_eager(condition: object, message: str = "") -> None:
    """Docstring."""
    record_assertion(condition, message)
