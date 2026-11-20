"""Eager mode implementations for control flow operations."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from types import ModuleType

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
    is_variadic_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap

_StateT = typing.TypeVar("_StateT")


def _index_cf_data(val: OpValue, index: int) -> OpValue:
    """Slice ``val[index]`` on an iterable backend array or sequence, or raise ``TypeError``.

    Args:
        val (OpValue): Backend array or sequence.
        index (int): Leading-axis index.

    Returns:
        OpValue: Element at ``index``.

    Raises:
        TypeError: If ``val`` is not iterable.
    """
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes, dict)):
        items: list[OpValue] = list(val)
        return items[index]
    raise TypeError(f"'{type(val).__name__}' object is not subscriptable")


def _ensure_cf_list(val: OpValue) -> list[OpValue]:
    """Convert an iterable or sequence `val` into `list[OpValue]`.

    Args:
        val (OpValue): Iterable or sequence value.

    Returns:
        list[OpValue]: Converted list.
    """
    return list(val) if isinstance(val, Iterable) else [val]


def cond_eager(pred: Tensor, true_fn: OpValue, false_fn: OpValue) -> OpValue:
    """Execute conditional branch eagerly based on ``bool(pred.data)``.

    Args:
        pred (Tensor): Boolean scalar predicate tensor.
        true_fn (OpValue): True-branch callable.
        false_fn (OpValue): False-branch callable.

    Returns:
        OpValue: Evaluated result of the selected branch.
    """
    if bool(pred.data):
        _callable_fn_1: OpValue = true_fn
        assert is_variadic_callable(_callable_fn_1)
        return _callable_fn_1()
    _callable_fn_2: OpValue = false_fn
    assert is_variadic_callable(_callable_fn_2)
    return _callable_fn_2()


def while_loop_eager(
    cond_fn: Callable[[_StateT], OpValue],
    body_fn: Callable[[_StateT], _StateT],
    init_val: _StateT,
) -> _StateT:
    """Execute the eager while loop eager operation on the active backend array.

    Args:
        cond_fn (Callable[[_StateT], OpValue]): Input `cond_fn` argument (cond fn).
        body_fn (Callable[[_StateT], _StateT]): Input `body_fn` argument (body fn).
        init_val (_StateT): Input `init_val` argument (init val).

    Returns:
        _StateT: Final loop state value.
    """
    val: _StateT = init_val
    res = cond_fn(val)
    while bool(extract_data(res)):
        val = body_fn(val)
        res = cond_fn(val)

    return val


def _stack_scan_outputs(
    ys: list[OpValue],
    init: OpValue,
    last_y: OpValue,
) -> Tensor:
    """Execute the eager stack scan outputs operation on the active backend array.

    Args:
        ys (list[OpValue]): Input `ys` argument (ys).
        init (OpValue): Input `init` argument (init).
        last_y (OpValue): Input `last_y` argument (last y).

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
    f: OpValue,
    init: OpValue,
    xs: OpValue,
    length: int | None = None,
) -> tuple[OpValue, Tensor]:
    """Execute the eager scan operation over leading axis of ``xs``.

    Args:
        f (OpValue): Scan step callable ``(carry, x) -> (carry, y)``.
        init (OpValue): Initial carry state.
        xs (OpValue): Sequence tensor scanned along axis 0.
        length (int | None): Optional scan iteration count.

    Returns:
        tuple[OpValue, Tensor]: Final ``(carry, stacked_ys)`` pair.
    """
    fn_callable = f
    carry: OpValue = init
    ys: list[OpValue] = []
    xs_shape = extract_shape(xs)
    scan_length = length if length is not None else (xs_shape[0] if xs is not None and xs_shape else 0)
    y: OpValue = init
    for i in range(scan_length):
        x = (
            Tensor(
                _index_cf_data(extract_data(xs), i), TensorConfig(xs_shape[1:], extract_dtype(xs), extract_device(xs))
            )
            if xs is not None
            else None
        )
        _callable_fn_3: OpValue = fn_callable
        assert is_variadic_callable(_callable_fn_3)
        carry, y = _ensure_cf_list(_callable_fn_3(carry, x))
        ys.append(extract_data(y))

    out_tensor = _stack_scan_outputs(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _map_fn_eager_get_length(elems: Tensor | None) -> int:
    """Execute the eager map fn eager get length operation on the active backend array.

    Args:
        elems (Tensor | None): Input `elems` argument (elems).

    Returns:
        int: Computed `int` output.
    """
    if elems is None:
        return 0
    el_shape = extract_shape(elems)
    return el_shape[0] if len(el_shape) > 0 else 0


def _map_fn_eager_execute(fn: OpValue, elems: Tensor, length: int) -> list[OpValue]:
    """Execute the eager map fn eager execute operation on the active backend array.

    Args:
        fn (OpValue): Input `fn` argument (fn).
        elems (Tensor): Input `elems` argument (elems).
        length (int): Input `length` argument (length).

    Returns:
        list: Computed `list` output.
    """
    fn_callable = fn
    ys: list[OpValue] = []
    for i in range(length):
        x = Tensor(_index_cf_data(elems.data, i), TensorConfig(elems.shape[1:], elems.dtype, elems.device))
        _callable_fn_4: OpValue = fn_callable
        assert is_variadic_callable(_callable_fn_4)
        y = _callable_fn_4(x)
        ys.append(extract_data(y))
    return ys


def _map_fn_eager_stack(ys: OpValue, elems: Tensor, dtype: DType | None) -> OpValue:
    """Execute the eager map fn eager stack operation on the active backend array.

    Args:
        ys (OpValue): Input `ys` argument (ys).
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


def map_fn_eager(fn: OpValue, elems: Tensor, dtype: DType | None = None) -> OpValue:
    """Execute the eager map fn eager operation on the active backend array.

    Args:
        fn (OpValue): Input `fn` argument (fn).
        elems (Tensor): Input `elems` argument (elems).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    length = _map_fn_eager_get_length(elems)
    ys = _map_fn_eager_execute(fn, elems, length)
    return _map_fn_eager_stack(ys, elems, dtype)


def pmap_eager(func: OpValue, axis_name: str | None = None) -> OpValue:
    """Execute the eager pmap eager operation on the active backend array.

    Args:
        func (OpValue): Input `func` argument (func).
        axis_name (str | None): Input `axis_name` argument (axis name).

    Returns:
        Callable: Computed `Callable` output.
    """
    del axis_name

    def wrapped(*args: OpValue) -> OpValue:
        """Execute the eager wrapped operation on the active backend array.

        Args:
            *args (OpValue): Positional args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        assert callable(func) and not isinstance(func, ModuleType)
        _callable_fn_5: OpValue = vmap(func)
        assert is_variadic_callable(_callable_fn_5)
        return _callable_fn_5(*args)

    return wrapped


def stop_gradient_eager(x: _StateT) -> _StateT:
    """Execute the eager stop gradient eager operation on the active backend array.

    Args:
        x (_StateT): Input `x` argument (x).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return x


def assert_value_eager(condition: OpValue, message: str = "") -> None:
    """Execute the eager assert value eager operation on the active backend array.

    Args:
        condition (OpValue): Input `condition` argument (condition).
        message (str): Input `message` argument (message).
    """
    record_assertion(condition, message)
