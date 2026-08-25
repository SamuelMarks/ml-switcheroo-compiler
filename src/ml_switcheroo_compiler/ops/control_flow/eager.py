"""Module eager.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Eager mode implementations for control flow operations."""


from typing import Callable

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap


def cond_eager(pred: Tensor, true_fn: Callable[[], object], false_fn: Callable[[], object]) -> object:
    """Evaluate cond_eager operation.

    Args:
        pred (Tensor): The pred parameter.
        true_fn (Callable): The true_fn parameter.
        false_fn (Callable): The false_fn parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if bool(pred.data):
        return true_fn()
    return false_fn()


def while_loop_eager(cond_fn: Callable[[object], Tensor], body_fn: Callable[[object], object], init_val: object) -> object:
    """Evaluate while_loop_eager operation.

    Args:
        cond_fn (object): The cond_fn parameter.
        body_fn (object): The body_fn parameter.
        init_val (object): The init_val parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    val: object = init_val
    res: object = cond_fn(val)
    while bool(res.data if hasattr(res, "data") else res):
        val: object = body_fn(val)
        res: object = cond_fn(val)

    return val


def _stack_scan_outputs(ys: list[object], init: object, last_y: object) -> object:
    """Evaluate _stack_scan_outputs operation.

    Args:
        ys (list): The ys parameter.
        init (object): The init parameter.
        last_y (object): The last_y parameter.

    Returns:
        Tensor: Result.
    """
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys: object = get_active_backend().execute_op("Stack", ys)
        return Tensor(
            stacked_ys,
            TensorConfig(
                stacked_ys.shape,
                last_y.dtype if hasattr(last_y, "dtype") else init.dtype,
                last_y.device if hasattr(last_y, "device") else init.device,
            ),
        )
    else:
        stacked_ys: object = get_active_backend().array(ys)
        return Tensor(
            stacked_ys,
            TensorConfig(stacked_ys.shape, DType(str(stacked_ys.dtype)), config.default_device),
        )


def scan_eager(f: Callable[..., object], init: object, xs: object, length: int | None = None) -> tuple[object, object]:
    """Evaluate scan_eager operation.

    Args:
        f (Callable): The f parameter.
        init (object): The init parameter.
        xs (object): The xs parameter.
        length (object): The length parameter.

    Returns:
        tuple: Result.
    """
    carry: object = init
    ys: object = []
    scan_length: object = length if length is not None else (xs.shape[0] if xs is not None else 0)
    for i in range(scan_length):
        x: object = Tensor(xs.data[i], TensorConfig(xs.shape[1:], xs.dtype, xs.device)) if xs is not None else None
        carry, y = f(carry, x)
        ys.append(y.data if hasattr(y, "data") else y)

    out_tensor: object = _stack_scan_outputs(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _map_fn_eager_get_length(elems: Tensor) -> int:
    """Evaluate _map_fn_eager_get_length operation.

    Args:
        elems (Tensor): The elems parameter.

    Returns:
        int: Result.
    """
    return elems.shape[0] if elems is not None and len(elems.shape) > 0 else 0


def _map_fn_eager_execute(fn: Callable[..., object], elems: Tensor, length: int) -> list[object]:
    """Evaluate _map_fn_eager_execute operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        length (int): The length parameter.

    Returns:
        list: Result.
    """
    ys: object = []
    for i in range(length):
        x: object = Tensor(elems.data[i], TensorConfig(elems.shape[1:], elems.dtype, elems.device))
        y: object = fn(x)
        ys.append(y.data if hasattr(y, "data") else y)
    return ys


def _map_fn_eager_stack(ys: list[object], elems: Tensor, dtype: DType | None) -> object:
    """Evaluate _map_fn_eager_stack operation.

    Args:
        ys (list): The ys parameter.
        elems (Tensor): The elems parameter.
        dtype (object): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys: object = get_active_backend().execute_op("Stack", ys)
        return Tensor(stacked_ys, TensorConfig(stacked_ys.shape, elems.dtype, elems.device))

    stacked_ys: object = get_active_backend().array(ys)
    out_dtype: object = dtype if dtype is not None else DType(str(stacked_ys.dtype))
    return Tensor(stacked_ys, TensorConfig(stacked_ys.shape, out_dtype, elems.device))


def map_fn_eager(fn: Callable[..., object], elems: Tensor, dtype: DType | None = None) -> object:
    """Evaluate map_fn_eager operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (object): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    length: object = _map_fn_eager_get_length(elems)
    ys: object = _map_fn_eager_execute(fn, elems, length)
    return _map_fn_eager_stack(ys, elems, dtype)


def pmap_eager(func: Callable[..., object], axis_name: str | None = None) -> Callable[..., object]:
    """Evaluate pmap_eager operation.

    Args:
        func (Callable): The func parameter.
        axis_name (object): The axis_name parameter.

    Returns:
        Callable: Result.
    """

    def wrapped(*args: object) -> object:
        """Evaluate wrapped operation.

        Args:
        *args (object): Positional args.

        Returns:
            tuple[int, ...]: Result.
        """
        return vmap(func)(*args)

    return wrapped


def stop_gradient_eager(x: object) -> object:
    """Evaluate stop_gradient_eager operation.

    Args:
        x (object): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


def assert_value_eager(condition: object, message: str = "") -> None:
    """Evaluate assert_value_eager operation.

    Args:
        condition (object): The condition parameter.
        message (str): The message parameter.
    """
    record_assertion(condition, message)
