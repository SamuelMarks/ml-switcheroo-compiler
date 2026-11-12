"""Eager mode implementations for control flow operations."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
    invoke_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap

_StateT = typing.TypeVar("_StateT")


def _is_cf_eager_tensor(val: TensorOrScalar) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        bool: True if non-None or None.
    """
    return not isinstance(val, type(Ellipsis))


def _ensure_cf_tensor(val: TensorOrScalar) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        EagerTensorProtocol: Narrowed tensor.
    """
    assert _is_cf_eager_tensor(val)
    return val


def _ensure_cf_list(val: TensorOrScalar) -> list[TensorOrScalar]:
    """Convert an iterable or sequence `val` into `list[TensorOrScalar]`.

    Args:
        val (TensorOrScalar): Iterable or sequence value.

    Returns:
        list[TensorOrScalar]: Converted list.
    """
    return list(val) if isinstance(val, Iterable) else [val]


def cond_eager(pred: Tensor, true_fn: TensorOrScalar, false_fn: TensorOrScalar) -> TensorOrScalar:
    """Execute the eager cond eager operation on the active backend array.

    Args:
        pred (Tensor): Input `pred` argument (pred).
        true_fn (TensorOrScalar): Input `true_fn` argument (true fn).
        false_fn (TensorOrScalar): Input `false_fn` argument (false fn).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if bool(pred.data):
        return invoke_callable(true_fn)
    return invoke_callable(false_fn)


def while_loop_eager(
    cond_fn: Callable[[_StateT], bool | Tensor],
    body_fn: Callable[[_StateT], _StateT],
    init_val: _StateT,
) -> _StateT:
    """Execute the eager while loop eager operation on the active backend array.

    Args:
        cond_fn (Callable[[_StateT], bool | Tensor]): Input `cond_fn` argument (cond fn).
        body_fn (Callable[[_StateT], _StateT]): Input `body_fn` argument (body fn).
        init_val (_StateT): Input `init_val` argument (init val).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    val = init_val
    res = cond_fn(val)
    while bool(extract_data(res)):
        val = body_fn(val)
        res = cond_fn(val)

    return val


def _stack_scan_outputs(
    ys: list[TensorOrScalar],
    init: TensorOrScalar,
    last_y: TensorOrScalar,
) -> Tensor:
    """Execute the eager stack scan outputs operation on the active backend array.

    Args:
        ys (list): Input `ys` argument (ys).
        init (TensorOrScalar): Input `init` argument (init).
        last_y (TensorOrScalar): Input `last_y` argument (last y).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(
            stacked_ys,
            TensorConfig(
                extract_shape(stacked_ys),
                extract_dtype(last_y, extract_dtype(init)),
                extract_device(last_y, extract_device(init)),
            ),
        )
    else:
        stacked_ys = get_active_backend().array(ys)
        return Tensor(
            stacked_ys,
            TensorConfig(extract_shape(stacked_ys), DType(str(extract_dtype(stacked_ys))), config.default_device),
        )


def scan_eager(
    f: TensorOrScalar,
    init: TensorOrScalar,
    xs: TensorOrScalar,
    length: int | None = None,
) -> TensorOrScalar:
    """Execute the eager scan eager operation on the active backend array.

    Args:
        f (TensorOrScalar): Input `f` argument (f).
        init (TensorOrScalar): Input `init` argument (init).
        xs (TensorOrScalar): Input `xs` argument (xs).
        length (int | None): Input `length` argument (length).

    Returns:
        tuple: Computed `tuple` output.
    """
    fn_callable = f
    carry: TensorOrScalar = init
    ys: list[TensorOrScalar] = []
    xs_shape = extract_shape(xs)
    scan_length = length if length is not None else (xs_shape[0] if xs is not None and xs_shape else 0)
    y: TensorOrScalar = init
    for i in range(scan_length):
        x = Tensor(_ensure_cf_tensor(extract_data(xs))[i], TensorConfig(xs_shape[1:], extract_dtype(xs), extract_device(xs))) if xs is not None else None
        carry, y = _ensure_cf_list(invoke_callable(fn_callable, carry, x))
        ys.append(extract_data(y))

    out_tensor = _stack_scan_outputs(ys, init, y if scan_length > 0 else init)
    return _ensure_cf_tensor(carry), out_tensor


def _map_fn_eager_get_length(elems: Tensor | None) -> int:
    """Execute the eager map fn eager get length operation on the active backend array.

    Args:
        elems (Tensor): Input `elems` argument (elems).

    Returns:
        int: Computed `int` output.
    """
    return elems.shape[0] if elems is not None and len(elems.shape) > 0 else 0


def _map_fn_eager_execute(fn: TensorOrScalar, elems: Tensor, length: int) -> list[TensorOrScalar]:
    """Execute the eager map fn eager execute operation on the active backend array.

    Args:
        fn (TensorOrScalar): Input `fn` argument (fn).
        elems (Tensor): Input `elems` argument (elems).
        length (int): Input `length` argument (length).

    Returns:
        list: Computed `list` output.
    """
    fn_callable = fn
    ys: list[TensorOrScalar] = []
    for i in range(length):
        x = Tensor(_ensure_cf_tensor(elems.data)[i], TensorConfig(elems.shape[1:], elems.dtype, elems.device))
        y = invoke_callable(fn_callable, x)
        ys.append(extract_data(y))
    return ys


def _map_fn_eager_stack(ys: TensorOrScalar, elems: Tensor, dtype: DType | None) -> TensorOrScalar:
    """Execute the eager map fn eager stack operation on the active backend array.

    Args:
        ys (list): Input `ys` argument (ys).
        elems (Tensor): Input `elems` argument (elems).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    ys_list = ys if isinstance(ys, (list, tuple)) else _ensure_cf_list(ys)
    if len(ys_list) > 0 and isinstance(ys_list[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys_list)
        return Tensor(stacked_ys, TensorConfig(extract_shape(stacked_ys), elems.dtype, elems.device))

    stacked_ys = get_active_backend().array(ys_list)
    out_dtype = dtype if dtype is not None else DType(str(extract_dtype(stacked_ys)))
    return Tensor(stacked_ys, TensorConfig(extract_shape(stacked_ys), out_dtype, elems.device))


def map_fn_eager(fn: TensorOrScalar, elems: Tensor, dtype: DType | None = None) -> TensorOrScalar:
    """Execute the eager map fn eager operation on the active backend array.

    Args:
        fn (TensorOrScalar): Input `fn` argument (fn).
        elems (Tensor): Input `elems` argument (elems).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    length = _map_fn_eager_get_length(elems)
    ys = _map_fn_eager_execute(fn, elems, length)
    return _map_fn_eager_stack(ys, elems, dtype)


def pmap_eager(func: TensorOrScalar, axis_name: str | None = None) -> TensorOrScalar:
    """Execute the eager pmap eager operation on the active backend array.

    Args:
        func (Callable): Input `func` argument (func).
        axis_name (str | None): Input `axis_name` argument (axis name).

    Returns:
        Callable: Computed `Callable` output.
    """
    del axis_name

    def wrapped(*args: TensorOrScalar) -> TensorOrScalar:
        """Execute the eager wrapped operation on the active backend array.

        Args:
            *args (TensorOrScalar): Positional args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return invoke_callable(vmap(func), *args)

    return wrapped


def stop_gradient_eager(x: _StateT) -> _StateT:
    """Execute the eager stop gradient eager operation on the active backend array.

    Args:
        x (_StateT): Input `x` argument (x).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return x


def assert_value_eager(condition: TensorOrScalar, message: str = "") -> None:
    """Execute the eager assert value eager operation on the active backend array.

    Args:
        condition (TensorOrScalar): Input `condition` argument (condition).
        message (str): Input `message` argument (message).
    """
    record_assertion(condition, message)
