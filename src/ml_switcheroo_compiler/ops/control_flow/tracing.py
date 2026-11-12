"""Tracing implementations for control flow operations."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import ensure_eager_list
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    TracedDataProtocol,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
    is_traced_data,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRNode, ensure_attrs
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state

_StateT = typing.TypeVar("_StateT")


def _as_traced_data(val: TensorOrScalar) -> TracedDataProtocol:
    """Narrow `val` to `TracedDataProtocol` for `.id` access.

    Args:
        val (TensorOrScalar): Traced tensor data or proxy.

    Returns:
        TracedDataProtocol: Narrowed traced data object.
    """

    def _guard(cand: TensorOrScalar) -> typing.TypeGuard[TracedDataProtocol]:
        """Type guard narrowing `cand` to `TracedDataProtocol`.

        Args:
            cand (TensorOrScalar): Candidate value.

        Returns:
            bool: True for traced data payloads.
        """
        return not isinstance(cand, type(Ellipsis))

    assert _guard(val)
    return val


def _as_state_val(val: TensorOrScalar, ref: _StateT) -> _StateT:
    """Narrow `val` to `_StateT`.

    Args:
        val (TensorOrScalar): Traced state value.
        ref (_StateT): Reference state value for type inference.

    Returns:
        _StateT: Narrowed state value.
    """
    del ref

    def _guard(cand: TensorOrScalar) -> typing.TypeGuard[_StateT]:
        """Type guard narrowing `cand` to `_StateT`.

        Args:
            cand (TensorOrScalar): Candidate value.

        Returns:
            bool: True for state values.
        """
        return not isinstance(cand, type(Ellipsis))

    assert _guard(val)
    return val


def cond_tracing(pred: Tensor, true_fn: TensorOrScalar, false_fn: TensorOrScalar) -> TensorOrScalar:
    """Emit a symbolic `If` control-flow node for `cond`.

    Args:
        pred (Tensor): Input `pred` argument (pred).
        true_fn (TensorOrScalar): Input `true_fn` argument (true fn).
        false_fn (TensorOrScalar): Input `false_fn` argument (false fn).

    Returns:
        TensorOrScalar: Traced output tensor.

    Raises:
        TracingError: If called outside a tracing context.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Cond node outside of a tracing context.")
    true_graph = _trace_function(true_fn, (), "true_branch")
    false_graph = _trace_function(false_fn, (), "false_branch")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="If",
        inputs=[_as_traced_data(pred.data).id],
        attributes=ensure_attrs({"then_branch": true_graph, "else_branch": false_graph}),
        subgraphs={"then_branch": true_graph, "else_branch": false_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")
    return Tensor(proxy, TensorConfig((), DType.Float32, pred.device))


def while_loop_tracing(
    cond_fn: Callable[[_StateT], bool | Tensor],
    body_fn: Callable[[_StateT], _StateT],
    init_val: _StateT,
) -> _StateT:
    """Emit a symbolic `Loop` node for `while_loop`.

    Args:
        cond_fn (Callable[[_StateT], bool | Tensor]): Input `cond_fn` argument (cond fn).
        body_fn (Callable[[_StateT], _StateT]): Input `body_fn` argument (body fn).
        init_val (_StateT): Input `init_val` argument (init val).

    Returns:
        _StateT: Traced loop state output.

    Raises:
        TracingError: If called outside a tracing context.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit While node outside of a tracing context.")
    raw_init: TensorOrScalar = init_val if isinstance(init_val, (Tensor, list, tuple)) else []
    args: tuple[TensorOrScalar, ...] = (init_val,) if isinstance(init_val, Tensor) else tuple(ensure_eager_list(raw_init))
    cond_graph = _trace_function(cond_fn, args, "cond")
    body_graph = _trace_function(body_fn, args, "body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Loop",
        inputs=[_as_traced_data(extract_data(a)).id for a in args],
        attributes=ensure_attrs({"body": body_graph, "cond": cond_graph}),
        subgraphs={"body": body_graph, "cond": cond_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    if isinstance(init_val, Tensor):
        proxy = ProxyTensor(id=out_id, shape=init_val.shape, dtype=init_val.dtype.value)
        return _as_state_val(Tensor(proxy, TensorConfig(init_val.shape, init_val.dtype, init_val.device)), init_val)
    return init_val


def _flatten_inputs(obj: TensorOrScalar) -> list[str]:
    """Flatten nested tensor arguments into a list of node IDs.

    Args:
        obj (TensorOrScalar): Input `obj` argument (obj).

    Returns:
        list[str]: Flattened node IDs.
    """
    if isinstance(obj, Tensor):
        return [_as_traced_data(obj.data).id]
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(_flatten_inputs(item))
        return res
    return []


def scan_tracing(
    f: TensorOrScalar,
    init: TensorOrScalar,
    xs: TensorOrScalar,
    length: int | None = None,
) -> TensorOrScalar:
    """Emit a symbolic `Scan` node for `scan`.

    Args:
        f (TensorOrScalar): Input `f` argument (f).
        init (TensorOrScalar): Input `init` argument (init).
        xs (TensorOrScalar): Input `xs` argument (xs).
        length (int | None): Input `length` argument (length).

    Returns:
        TensorOrScalar: Tuple of `(carry, stacked_outputs)`.

    Raises:
        TracingError: If called outside a tracing context.
    """
    _ = length
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Scan node outside of a tracing context.")
    xs_shape = extract_shape(xs)
    xs_dtype = extract_dtype(xs)
    xs_dtype_str = xs_dtype.value if isinstance(xs_dtype, DType) else str(xs_dtype)
    xs_device = extract_device(xs)
    x_shape = xs_shape[1:] if xs is not None and len(xs_shape) > 0 else ()
    proxy_x = ProxyTensor(id="proxy_x_tensor", shape=x_shape, dtype=xs_dtype_str)
    proxy_x_tensor = Tensor(proxy_x, TensorConfig(x_shape, xs_dtype, xs_device))
    from ml_switcheroo_compiler.backends.registry import ensure_eager_tensor

    body_graph = _trace_function(ensure_eager_tensor(f), (ensure_eager_tensor(init), proxy_x_tensor), "scan_body")
    init_ids = _flatten_inputs(init)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=init_ids + [_as_traced_data(extract_data(xs)).id],
        attributes=ensure_attrs({"body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=xs_shape, dtype=xs_dtype_str)
    out_tensor = Tensor(proxy, TensorConfig(xs_shape, xs_dtype, xs_device))
    return ensure_eager_tensor(init), out_tensor


def map_fn_tracing(fn: TensorOrScalar, elems: Tensor, dtype: DType | None = None) -> TensorOrScalar:
    """Emit a symbolic `Map` node for `map_fn`.

    Args:
        fn (TensorOrScalar): Input `fn` argument (fn).
        elems (Tensor): Input `elems` argument (elems).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        TensorOrScalar: Mapped output tensor.

    Raises:
        TracingError: If called outside a tracing context.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Map node outside of a tracing context.")
    x_shape = elems.shape[1:] if len(elems.shape) > 0 else ()
    proxy_x = ProxyTensor(id="proxy_x_tensor", shape=x_shape, dtype=elems.dtype.value)
    proxy_x_tensor = Tensor(proxy_x, TensorConfig(x_shape, elems.dtype, elems.device))
    body_graph = _trace_function(fn, (proxy_x_tensor,), "map_body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Map",
        inputs=[_as_traced_data(elems.data).id],
        attributes=ensure_attrs({"body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    out_dtype = dtype if dtype is not None else elems.dtype
    out_shape = (elems.shape[0],)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, elems.device))


def pmap_tracing(func: TensorOrScalar, axis_name: str | None = None) -> TensorOrScalar:
    """Create a symbolic `Pmap` wrapper for `func`.

    Args:
        func (TensorOrScalar): Input `func` argument (func).
        axis_name (str | None): Input `axis_name` argument (axis name).

    Returns:
        TensorOrScalar: Wrapped parallel-map callable.
    """

    def wrapped(*args: TensorOrScalar) -> TensorOrScalar:
        """Trace `func` and emit a `Pmap` node.

        Args:
            *args (TensorOrScalar): Positional args.

        Returns:
            TensorOrScalar: Traced output tensor.

        Raises:
            TracingError: If called outside a tracing context.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError("Cannot emit Pmap outside of a tracing context.")
        proxy_args: list[TensorOrScalar] = []
        for a in args:
            if isinstance(a, Tensor):
                new_shape = a.shape[1:] if len(a.shape) > 0 else ()
                proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
                proxy_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
            else:
                proxy_args.append(a)
        body_graph = _trace_function(func, tuple(proxy_args), "pmap_body")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Pmap",
            inputs=[str(a.data.id) if is_traced_data(a.data) else "" for a in args if isinstance(a, Tensor)],
            attributes=ensure_attrs({"axis_name": axis_name, "body": body_graph}),
            subgraphs={"body": body_graph},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        arg = args[0]
        arg_shape = extract_shape(arg)
        arg_dtype = extract_dtype(arg)
        arg_dtype_str = arg_dtype.value if isinstance(arg_dtype, DType) else str(arg_dtype)
        proxy = ProxyTensor(id=out_id, shape=arg_shape, dtype=arg_dtype_str)
        return Tensor(proxy, TensorConfig(arg_shape, arg_dtype, extract_device(arg)))

    return wrapped


def stop_gradient_tracing(x: _StateT) -> _StateT:
    """Compute the stop gradient tracing tensor operation.

    Args:
        x (_StateT): Input `x` argument (x).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if not global_tracing_state.is_tracing:
        return x
    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.data.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
        return _as_state_val(Tensor(proxy, TensorConfig(x.shape, x.dtype, x.device)), x)
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        return _as_state_val(ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype), x)
    return x


def assert_value_tracing(condition: TensorOrScalar, message: str = "") -> None:
    """Compute the assert value tracing tensor operation.

    Args:
        condition (TensorOrScalar): Input `condition` argument (condition).
        message (str): Input `message` argument (message).

    Returns:
        NoneType: Computed `NoneType` output.
    """
    if not global_tracing_state.is_tracing:
        record_assertion(condition, message)
        return
    inp_id = _as_traced_data(condition.data if isinstance(condition, Tensor) else condition).id
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Assert",
        inputs=[inp_id],
        attributes={"message": message},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
