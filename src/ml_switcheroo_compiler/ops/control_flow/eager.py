"""Eager mode implementations for control flow operations."""

from __future__ import annotations

from typing import Callable, TypeVar, cast  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue, get_active_backend
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap

_StateT = TypeVar("_StateT")


def cond_eager(pred: Tensor, true_fn: Callable[..., EagerValue], false_fn: Callable[..., EagerValue]) -> EagerValue:
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


def while_loop_eager(
    cond_fn: Callable[[_StateT], bool | Tensor],
    body_fn: Callable[[_StateT], _StateT],
    init_val: _StateT,
) -> _StateT:
    """Evaluate while_loop_eager operation.

    Args:
        cond_fn (Any): The cond_fn parameter.
        body_fn (Any): The body_fn parameter.
        init_val (Any): The init_val parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    val = init_val
    res = cond_fn(val)
    while bool(res.data if hasattr(res, "data") else res):
        val = body_fn(val)
        res = cond_fn(val)

    return val


def _stack_scan_outputs(
    ys: list[EagerValue],
    init: Tensor | tuple[Tensor, ...],
    last_y: Tensor | tuple[Tensor, ...],
) -> Tensor:
    """Evaluate _stack_scan_outputs operation.

    Args:
        ys (list): The ys parameter.
        init (Any): The init parameter.
        last_y (Any): The last_y parameter.

    Returns:
        Tensor: Result.
    """
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(
            stacked_ys,
            TensorConfig(
                getattr(stacked_ys, "shape"),
                getattr(last_y, "dtype") if hasattr(last_y, "dtype") else getattr(init, "dtype"),
                getattr(last_y, "device") if hasattr(last_y, "device") else getattr(init, "device"),
            ),
        )
    else:
        stacked_ys = get_active_backend().array(ys)
        return Tensor(
            stacked_ys,
            TensorConfig(getattr(stacked_ys, "shape"), DType(str(getattr(stacked_ys, "dtype"))), config.default_device),
        )


def scan_eager(f: Callable[..., EagerValue], init: EagerValue, xs: EagerValue, length: int | None = None) -> EagerValue:
    """Evaluate scan_eager operation.

    Args:
        f (Callable): The f parameter.
        init (Any): The init parameter.
        xs (Any): The xs parameter.
        length (Any): The length parameter.

    Returns:
        tuple: Result.
    """
    carry = init
    ys = []
    scan_length = length if length is not None else (getattr(xs, "shape")[0] if xs is not None else 0)
    for i in range(scan_length):
        x = Tensor(getattr(xs, "data")[i], TensorConfig(getattr(xs, "shape")[1:], getattr(xs, "dtype"), getattr(xs, "device"))) if xs is not None else None
        carry, y = getattr(f(carry, x), "__iter__")()
        ys.append(getattr(y, "data") if hasattr(y, "data") else y)

    out_tensor = cast("Callable[..., EagerTensorProtocol]", _stack_scan_outputs)(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _map_fn_eager_get_length(elems: Tensor) -> int:
    """Evaluate _map_fn_eager_get_length operation.

    Args:
        elems (Tensor): The elems parameter.

    Returns:
        int: Result.
    """
    return elems.shape[0] if elems is not None and len(elems.shape) > 0 else 0


def _map_fn_eager_execute(fn: Callable[..., EagerValue], elems: Tensor, length: int) -> EagerValue:
    """Evaluate _map_fn_eager_execute operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        length (int): The length parameter.

    Returns:
        list: Result.
    """
    ys = []
    for i in range(length):
        x = Tensor(cast(EagerTensorProtocol, elems.data)[i], TensorConfig(elems.shape[1:], elems.dtype, elems.device))
        y = fn(x)
        ys.append(getattr(y, "data") if hasattr(y, "data") else y)
    return ys


def _map_fn_eager_stack(ys: EagerValue, elems: Tensor, dtype: DType | None) -> EagerValue:
    """Evaluate _map_fn_eager_stack operation.

    Args:
        ys (list): The ys parameter.
        elems (Tensor): The elems parameter.
        dtype (Any): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if getattr(ys, "__len__")() > 0 and isinstance(getattr(ys, "__getitem__")(0), tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(stacked_ys, TensorConfig(getattr(stacked_ys, "shape"), elems.dtype, elems.device))

    stacked_ys = get_active_backend().array(ys)
    out_dtype = dtype if dtype is not None else DType(str(getattr(stacked_ys, "dtype")))
    return Tensor(stacked_ys, TensorConfig(getattr(stacked_ys, "shape"), out_dtype, elems.device))


def map_fn_eager(fn: Callable[..., EagerValue], elems: Tensor, dtype: DType | None = None) -> EagerValue:
    """Evaluate map_fn_eager operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (Any): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    length = _map_fn_eager_get_length(elems)
    ys = _map_fn_eager_execute(fn, elems, length)
    return _map_fn_eager_stack(ys, elems, dtype)


def pmap_eager(func: Callable[..., EagerValue], axis_name: str | None = None) -> EagerValue:
    """Evaluate pmap_eager operation.

    Args:
        func (Callable): The func parameter.
        axis_name (Any): The axis_name parameter.

    Returns:
        Callable: Result.
    """

    def wrapped(*args: EagerValue) -> EagerValue:
        """Evaluate wrapped operation.

        Args:
        *args (Any): Positional args.

        Returns:
            tuple[int, ...]: Result.
        """
        return cast("Callable[..., EagerValue]", vmap(func))(*args)

    return wrapped


def stop_gradient_eager(x: _StateT) -> _StateT:
    """Evaluate stop_gradient_eager operation.

    Args:
        x (Any): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


def assert_value_eager(condition: EagerValue, message: str = "") -> None:
    """Evaluate assert_value_eager operation.

    Args:
        condition (Any): The condition parameter.
        message (str): The message parameter.
    """
    record_assertion(condition, message)
