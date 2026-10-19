"""Eager mode implementations for control flow operations."""

from __future__ import annotations

from typing import Callable, TypeVar, cast  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerCallableProtocol, EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap

_StateT = TypeVar("_StateT")


def cond_eager(pred: Tensor, true_fn: Callable[..., TensorOrScalar], false_fn: Callable[..., TensorOrScalar]) -> TensorOrScalar:
    """Execute the eager cond eager operation on the active backend array.

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
    """Execute the eager while loop eager operation on the active backend array.

    Args:
        cond_fn (Callable[[_StateT], bool | Tensor]): The cond_fn parameter.
        body_fn (Callable[[_StateT], _StateT]): The body_fn parameter.
        init_val (_StateT): The init_val parameter.

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
    ys: list[TensorOrScalar],
    init: Tensor | tuple[Tensor, ...],
    last_y: Tensor | tuple[Tensor, ...],
) -> Tensor:
    """Execute the eager stack scan outputs operation on the active backend array.

    Args:
        ys (list): The ys parameter.
        init (Tensor | tuple[Tensor, ...]): The init parameter.
        last_y (Tensor | tuple[Tensor, ...]): The last_y parameter.

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


def scan_eager(f: Callable[..., TensorOrScalar], init: TensorOrScalar, xs: TensorOrScalar, length: int | None = None) -> TensorOrScalar:
    """Execute the eager scan eager operation on the active backend array.

    Args:
        f (Callable): The f parameter.
        init (TensorOrScalar): The init parameter.
        xs (TensorOrScalar): The xs parameter.
        length (int | None): The length parameter.

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

    out_tensor = cast("EagerCallableProtocol", _stack_scan_outputs)(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _map_fn_eager_get_length(elems: Tensor) -> int:
    """Execute the eager map fn eager get length operation on the active backend array.

    Args:
        elems (Tensor): The elems parameter.

    Returns:
        int: Result.
    """
    return elems.shape[0] if elems is not None and len(elems.shape) > 0 else 0


def _map_fn_eager_execute(fn: Callable[..., TensorOrScalar], elems: Tensor, length: int) -> TensorOrScalar:
    """Execute the eager map fn eager execute operation on the active backend array.

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


def _map_fn_eager_stack(ys: TensorOrScalar, elems: Tensor, dtype: DType | None) -> TensorOrScalar:
    """Execute the eager map fn eager stack operation on the active backend array.

    Args:
        ys (list): The ys parameter.
        elems (Tensor): The elems parameter.
        dtype (DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if getattr(ys, "__len__")() > 0 and isinstance(getattr(ys, "__getitem__")(0), tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(stacked_ys, TensorConfig(getattr(stacked_ys, "shape"), elems.dtype, elems.device))

    stacked_ys = get_active_backend().array(ys)
    out_dtype = dtype if dtype is not None else DType(str(getattr(stacked_ys, "dtype")))
    return Tensor(stacked_ys, TensorConfig(getattr(stacked_ys, "shape"), out_dtype, elems.device))


def map_fn_eager(fn: Callable[..., TensorOrScalar], elems: Tensor, dtype: DType | None = None) -> TensorOrScalar:
    """Execute the eager map fn eager operation on the active backend array.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (DType | None): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    length = _map_fn_eager_get_length(elems)
    ys = _map_fn_eager_execute(fn, elems, length)
    return _map_fn_eager_stack(ys, elems, dtype)


def pmap_eager(func: Callable[..., TensorOrScalar], axis_name: str | None = None) -> TensorOrScalar:
    """Execute the eager pmap eager operation on the active backend array.

    Args:
        func (Callable): The func parameter.
        axis_name (str | None): The axis_name parameter.

    Returns:
        Callable: Result.
    """

    def wrapped(*args: TensorOrScalar) -> TensorOrScalar:
        """Execute the eager wrapped operation on the active backend array.

        Args:
        *args (TensorOrScalar): Positional args.

        Returns:
            tuple[int, ...]: Result.
        """
        return cast("Callable[..., TensorOrScalar]", vmap(func))(*args)

    return wrapped


def stop_gradient_eager(x: _StateT) -> _StateT:
    """Execute the eager stop gradient eager operation on the active backend array.

    Args:
        x (_StateT): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return x


def assert_value_eager(condition: TensorOrScalar, message: str = "") -> None:
    """Execute the eager assert value eager operation on the active backend array.

    Args:
        condition (TensorOrScalar): The condition parameter.
        message (str): The message parameter.
    """
    record_assertion(condition, message)
