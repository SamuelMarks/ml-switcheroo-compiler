"""Tracing implementations for control flow operations."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRNode, as_attrs
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state

_StateT = typing.TypeVar("_StateT")


def cond_tracing(pred: Tensor, true_fn: TensorOrScalar, false_fn: TensorOrScalar) -> TensorOrScalar:
    """Emit a symbolic `If` control-flow node for `cond`.

    Args:
        pred (Tensor): The pred parameter.
        true_fn (TensorOrScalar): The true_fn parameter.
        false_fn (TensorOrScalar): The false_fn parameter.

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
        inputs=[getattr(pred.data, "id")],
        attributes=as_attrs({"then_branch": true_graph, "else_branch": false_graph}),
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
        cond_fn (Callable[[_StateT], bool | Tensor]): The cond_fn parameter.
        body_fn (Callable[[_StateT], _StateT]): The body_fn parameter.
        init_val (_StateT): The init_val parameter.

    Returns:
        _StateT: Traced loop state output.

    Raises:
        TracingError: If called outside a tracing context.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit While node outside of a tracing context.")
    args = (init_val,) if isinstance(init_val, Tensor) else tuple(getattr(init_val, "__iter__")())
    cond_graph = _trace_function(cond_fn, args, "cond")
    body_graph = _trace_function(body_fn, args, "body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Loop",
        inputs=[getattr(a.data, "id") for a in args],
        attributes=as_attrs({"body": body_graph, "cond": cond_graph}),
        subgraphs={"body": body_graph, "cond": cond_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    if isinstance(init_val, Tensor):
        proxy = ProxyTensor(id=out_id, shape=init_val.shape, dtype=init_val.dtype.value)
        return typing.cast(_StateT, Tensor(proxy, TensorConfig(init_val.shape, init_val.dtype, init_val.device)))
    return init_val


def _flatten_inputs(obj: TensorOrScalar) -> list[str]:
    """Flatten nested tensor arguments into a list of node IDs.

    Args:
        obj (TensorOrScalar): The obj parameter.

    Returns:
        list[str]: Flattened node IDs.
    """
    if isinstance(obj, Tensor):
        return [getattr(obj.data, "id")]
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(_flatten_inputs(item))
        return res
    return []


def scan_tracing(f: TensorOrScalar, init: TensorOrScalar, xs: TensorOrScalar, length: int | None = None) -> TensorOrScalar:
    """Emit a symbolic `Scan` node for `scan`.

    Args:
        f (TensorOrScalar): The f parameter.
        init (TensorOrScalar): The init parameter.
        xs (TensorOrScalar): The xs parameter.
        length (int | None): The length parameter.

    Returns:
        TensorOrScalar: Tuple of `(carry, stacked_outputs)`.

    Raises:
        TracingError: If called outside a tracing context.
    """
    _ = length
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Scan node outside of a tracing context.")
    x_shape = getattr(xs, "shape")[1:] if xs is not None and len(getattr(xs, "shape")) > 0 else ()
    proxy_x = ProxyTensor(id="proxy_x_tensor", shape=x_shape, dtype=getattr(getattr(xs, "dtype"), "value"))
    proxy_x_tensor = Tensor(proxy_x, TensorConfig(x_shape, getattr(xs, "dtype"), getattr(xs, "device")))
    body_graph = _trace_function(f, (init, proxy_x_tensor), "scan_body")
    init_ids = _flatten_inputs(init)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=init_ids + [getattr(getattr(xs, "data"), "id")],
        attributes=as_attrs({"body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=getattr(xs, "shape"), dtype=getattr(getattr(xs, "dtype"), "value"))
    out_tensor = Tensor(proxy, TensorConfig(getattr(xs, "shape"), getattr(xs, "dtype"), getattr(xs, "device")))
    return init, out_tensor


def map_fn_tracing(fn: TensorOrScalar, elems: Tensor, dtype: DType | None = None) -> TensorOrScalar:
    """Emit a symbolic `Map` node for `map_fn`.

    Args:
        fn (TensorOrScalar): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (DType | None): The dtype parameter.

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
        inputs=[getattr(elems.data, "id")],
        attributes=as_attrs({"body": body_graph}),
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
        func (TensorOrScalar): The func parameter.
        axis_name (str | None): The axis_name parameter.

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
        proxy_args = []
        for a in args:
            if isinstance(a, Tensor):
                new_shape = a.shape[1:] if len(a.shape) > 0 else ()
                proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
                proxy_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
            else:
                getattr(proxy_args, "append")(a)
        body_graph = _trace_function(func, tuple(proxy_args), "pmap_body")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Pmap",
            inputs=[str(getattr(getattr(a, "data", None), "id", "")) for a in args if isinstance(a, Tensor)],
            attributes=as_attrs({"axis_name": axis_name, "body": body_graph}),
            subgraphs={"body": body_graph},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        arg = args[0]
        proxy = ProxyTensor(id=out_id, shape=getattr(arg, "shape"), dtype=getattr(getattr(arg, "dtype"), "value"))
        return Tensor(proxy, TensorConfig(getattr(arg, "shape"), getattr(arg, "dtype"), getattr(arg, "device")))

    return wrapped


def stop_gradient_tracing(x: _StateT) -> _StateT:
    """Compute the stop gradient tracing tensor operation.

    Args:
        x (_StateT): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if not global_tracing_state.is_tracing:
        return x
    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.data.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
        return typing.cast(_StateT, Tensor(proxy, TensorConfig(x.shape, x.dtype, x.device)))
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        return typing.cast(_StateT, ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype))
    return x


def assert_value_tracing(condition: TensorOrScalar, message: str = "") -> None:
    """Compute the assert value tracing tensor operation.

    Args:
        condition (TensorOrScalar): The condition parameter.
        message (str): The message parameter.

    Returns:
        NoneType: Result.
    """
    if not global_tracing_state.is_tracing:
        record_assertion(condition, message)
        return
    inp_id = getattr(condition.data, "id") if isinstance(condition, Tensor) else getattr(condition, "id")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Assert",
        inputs=[inp_id],
        attributes={"message": message},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
