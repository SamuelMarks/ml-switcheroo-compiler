"""Tracing implementations for control flow operations."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Callable

from ml_switcheroo_ir import IRNode, LogicalNode

from ml_switcheroo_compiler.backends.registry import coerce_operand_list
from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
    is_traced_data,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state

_StateT = typing.TypeVar("_StateT")


@typing.runtime_checkable
class _TracedIdProtocol(typing.Protocol):
    """Protocol for traced proxy payloads exposing an ``id`` string."""

    @property
    def id(self) -> str:
        """Return traced node identifier.

        Returns:
            str: Traced node ID.
        """
        ...


def _extract_traced_id(val: OpValue) -> str:
    """Extract ``.id`` from a traced tensor payload or raise ``AttributeError``.

    Args:
        val (OpValue): Traced tensor data or proxy.

    Returns:
        str: Traced node identifier.

    Raises:
        AttributeError: If ``val`` does not expose an ``id`` attribute.
    """
    if isinstance(val, _TracedIdProtocol):
        return str(val.id)
    raise AttributeError(f"'{type(val).__name__}' object has no attribute 'id'")


def cond_tracing(pred: Tensor, true_fn: OpValue, false_fn: OpValue) -> Tensor:
    """Emit a symbolic `If` control-flow node for `cond`.

    Args:
        pred (Tensor): Input `pred` argument (pred).
        true_fn (OpValue): Input `true_fn` argument (true fn).
        false_fn (OpValue): Input `false_fn` argument (false fn).

    Returns:
        Tensor: Traced output tensor.

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
        inputs=[_extract_traced_id(pred.data)],
        attributes=ensure_attrs({"then_branch": true_graph, "else_branch": false_graph}),
        subgraphs={"then_branch": true_graph, "else_branch": false_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")
    return Tensor(proxy, TensorConfig((), DType.Float32, pred.device))


@typing.overload
def while_loop_tracing(
    cond_fn: Callable[[Tensor], OpValue],
    body_fn: Callable[[Tensor], Tensor],
    init_val: Tensor,
) -> Tensor: ...


@typing.overload
def while_loop_tracing(
    cond_fn: Callable[[_StateT], OpValue],
    body_fn: Callable[[_StateT], _StateT],
    init_val: _StateT,
) -> _StateT | Tensor: ...


def while_loop_tracing(
    cond_fn: Callable[[_StateT], OpValue],
    body_fn: Callable[[_StateT], _StateT],
    init_val: _StateT,
) -> _StateT | Tensor:
    """Emit a symbolic `Loop` node for `while_loop`.

    Args:
        cond_fn (Callable[[_StateT], OpValue]): Input `cond_fn` argument (cond fn).
        body_fn (Callable[[_StateT], _StateT]): Input `body_fn` argument (body fn).
        init_val (_StateT): Input `init_val` argument (init val).

    Returns:
        _StateT | Tensor: Traced loop state output.

    Raises:
        TracingError: If called outside a tracing context.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit While node outside of a tracing context.")
    raw_init: OpValue = init_val if isinstance(init_val, (Tensor, list, tuple)) else []
    args: tuple[OpValue, ...] = (init_val,) if isinstance(init_val, Tensor) else tuple(coerce_operand_list(raw_init))
    cond_graph = _trace_function(cond_fn, args, "cond")
    body_graph = _trace_function(body_fn, args, "body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Loop",
        inputs=[_extract_traced_id(extract_data(a)) for a in args],
        attributes=ensure_attrs({"body": body_graph, "cond": cond_graph}),
        subgraphs={"body": body_graph, "cond": cond_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    if isinstance(init_val, Tensor):
        proxy = ProxyTensor(id=out_id, shape=init_val.shape, dtype=init_val.dtype.value)
        return Tensor(proxy, TensorConfig(init_val.shape, init_val.dtype, init_val.device))
    return init_val


def _flatten_inputs(obj: OpValue) -> list[str]:
    """Flatten nested tensor arguments into a list of node IDs.

    Args:
        obj (OpValue): Input `obj` argument (obj).

    Returns:
        list[str]: Flattened node IDs.
    """
    if isinstance(obj, Tensor):
        return [_extract_traced_id(obj.data)]
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(_flatten_inputs(item))
        return res
    return []


def scan_tracing(
    f: OpValue,
    init: OpValue,
    xs: OpValue,
    length: int | None = None,
) -> tuple[OpValue, Tensor]:
    """Emit a symbolic `Scan` node for `scan`.

    Args:
        f (OpValue): Input `f` argument (f).
        init (OpValue): Input `init` argument (init).
        xs (OpValue): Input `xs` argument (xs).
        length (int | None): Input `length` argument (length).

    Returns:
        tuple[OpValue, Tensor]: Tuple of `(carry, stacked_outputs)`.

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

    body_graph = _trace_function(f, (init, proxy_x_tensor), "scan_body")
    init_ids = _flatten_inputs(init)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=init_ids + [_extract_traced_id(extract_data(xs))],
        attributes=ensure_attrs({"body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=xs_shape, dtype=xs_dtype_str)
    out_tensor = Tensor(proxy, TensorConfig(xs_shape, xs_dtype, xs_device))
    return init, out_tensor


def map_fn_tracing(fn: OpValue, elems: Tensor, dtype: DType | None = None) -> Tensor:
    """Emit a symbolic `Map` node for `map_fn`.

    Args:
        fn (OpValue): Input `fn` argument (fn).
        elems (Tensor): Input `elems` argument (elems).
        dtype (DType | None): Input `dtype` argument (dtype).

    Returns:
        Tensor: Mapped output tensor.

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
        inputs=[_extract_traced_id(elems.data)],
        attributes=ensure_attrs({"body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    out_dtype = dtype if dtype is not None else elems.dtype
    out_shape = (elems.shape[0],)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, elems.device))


def pmap_tracing(func: OpValue, axis_name: str | None = None) -> OpValue:
    """Create a symbolic `Pmap` wrapper for `func`.

    Args:
        func (OpValue): Input `func` argument (func).
        axis_name (str | None): Input `axis_name` argument (axis name).

    Returns:
        OpValue: Wrapped parallel-map callable.
    """

    def wrapped(*args: OpValue) -> OpValue:
        """Trace `func` and emit a `Pmap` node.

        Args:
            *args (OpValue): Positional args.

        Returns:
            OpValue: Traced output tensor.

        Raises:
            TracingError: If called outside a tracing context.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError("Cannot emit Pmap outside of a tracing context.")
        proxy_args: list[OpValue] = []
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


@typing.overload
def stop_gradient_tracing(x: Tensor) -> Tensor: ...


@typing.overload
def stop_gradient_tracing(x: ProxyTensor) -> ProxyTensor: ...


@typing.overload
def stop_gradient_tracing(x: _StateT) -> _StateT | Tensor | ProxyTensor: ...


def stop_gradient_tracing(x: _StateT) -> _StateT | Tensor | ProxyTensor:
    """Emit a symbolic ``StopGradient`` node when tracing, otherwise return ``x`` unchanged.

    Args:
        x (_StateT): Input tensor, proxy tensor, or state value.

    Returns:
        _StateT | Tensor | ProxyTensor: Stop-gradient wrapped tensor/proxy or ``x``.
    """
    if not global_tracing_state.is_tracing:
        return x
    from ml_switcheroo_compiler.ir.core import normalize_shape_metadata

    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id, op_type="StopGradient", inputs=[x.data.id], shape_metadata=normalize_shape_metadata(x.shape)
        )
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
        return Tensor(proxy, TensorConfig(x.shape, x.dtype, x.device))
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id, op_type="StopGradient", inputs=[x.id], shape_metadata=normalize_shape_metadata(x.shape)
        )
        global_tracing_state.add_node(node)
        return ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype)
    return x


def assert_value_tracing(condition: OpValue, message: str = "") -> None:
    """Compute the assert value tracing tensor operation.

    Args:
        condition (OpValue): Input `condition` argument (condition).
        message (str): Input `message` argument (message).

    Returns:
        NoneType: Computed `NoneType` output.
    """
    if not global_tracing_state.is_tracing:
        record_assertion(condition, message)
        return
    inp_id = _extract_traced_id(condition.data if isinstance(condition, Tensor) else condition)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Assert",
        inputs=[inp_id],
        attributes={"message": message},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
