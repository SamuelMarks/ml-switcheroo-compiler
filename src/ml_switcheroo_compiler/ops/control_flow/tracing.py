"""Tracing implementations for control flow operations."""

from __future__ import annotations

import typing
import uuid
from typing import Any, Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.assertions import record_assertion
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def cond_tracing(pred: Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> object:
    """Evaluate and process the cond tracing operation.

    Args:
        pred (Tensor): Required parameter for pred.
        true_fn (Callable): Required parameter for true_fn.
        false_fn (Callable): Required parameter for false_fn.

    Returns:
        object: The evaluated or processed output.
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
        inputs=[pred.data.id],
        attributes={"then_branch": true_graph, "else_branch": false_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")
    return Tensor(proxy, TensorConfig((), DType.Float32, pred.device))


def while_loop_tracing(cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: object) -> object:
    """Evaluate and process the while loop tracing operation.

    Args:
        cond_fn (Callable): Required parameter for cond_fn.
        body_fn (Callable): Required parameter for body_fn.
        init_val (object): Required parameter for init_val.

    Returns:
        object: The evaluated or processed output.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit While node outside of a tracing context.")
    args = (init_val,) if isinstance(init_val, Tensor) else tuple(typing.cast(typing.Iterable[Tensor], init_val))
    cond_graph = _trace_function(cond_fn, args, "cond")
    body_graph = _trace_function(body_fn, args, "body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Loop",
        inputs=[a.data.id for a in args],
        attributes={"body": body_graph, "cond": cond_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    if isinstance(init_val, Tensor):
        proxy = ProxyTensor(id=out_id, shape=init_val.shape, dtype=init_val.dtype.value)
        return Tensor(proxy, TensorConfig(init_val.shape, init_val.dtype, init_val.device))
    return init_val


def _flatten_inputs(obj: object) -> list[str]:
    """Evaluate and process the flatten inputs operation.

    Args:
        obj (object): Required parameter for obj.

    Returns:
        list: The evaluated or processed output.
    """
    if isinstance(obj, Tensor):
        return [obj.data.id]
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(_flatten_inputs(item))
        return res
    return []


def scan_tracing(f: Callable, init: object, xs: object, length: int | None = None) -> tuple[Any, Any]:
    """Evaluate and process the scan tracing operation.

    Args:
        f (Callable): Required parameter for f.
        init (object): Required parameter for init.
        xs (object): Required parameter for xs.
        length (Any): Required parameter for length.

    Returns:
        tuple: The evaluated or processed output.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Scan node outside of a tracing context.")
    x_shape = xs.shape[1:] if xs is not None and len(xs.shape) > 0 else ()
    proxy_x = ProxyTensor(id="dummy_x", shape=x_shape, dtype=xs.dtype.value)
    dummy_x = Tensor(proxy_x, TensorConfig(x_shape, xs.dtype, xs.device))
    body_graph = _trace_function(f, (init, dummy_x), "scan_body")
    init_ids = _flatten_inputs(init)
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=init_ids + [xs.data.id],
        attributes={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=xs.shape, dtype=xs.dtype.value)
    out_tensor = Tensor(proxy, TensorConfig(xs.shape, xs.dtype, xs.device))
    return init, out_tensor


def map_fn_tracing(fn: Callable, elems: Tensor, dtype: DType | None = None) -> Tensor:
    """Evaluate and process the map fn tracing operation.

    Args:
        fn (Callable): Required parameter for fn.
        elems (Tensor): Required parameter for elems.
        dtype (Any): Required parameter for dtype.

    Returns:
        Tensor: The evaluated or processed output.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Map node outside of a tracing context.")
    x_shape = elems.shape[1:] if elems is not None and len(elems.shape) > 0 else ()
    proxy_x = ProxyTensor(id="dummy_x", shape=x_shape, dtype=elems.dtype.value)
    dummy_x = Tensor(proxy_x, TensorConfig(x_shape, elems.dtype, elems.device))
    body_graph = _trace_function(fn, (dummy_x,), "map_body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Map",
        inputs=[elems.data.id],
        attributes={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    out_dtype = dtype if dtype is not None else elems.dtype
    out_shape = (elems.shape[0],)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, elems.device))


def pmap_tracing(func: Callable, axis_name: str | None = None) -> Callable:
    """Evaluate and process the pmap tracing operation.

    Args:
        func (Callable): Required parameter for func.
        axis_name (Any): Required parameter for axis_name.

    Returns:
        Callable: The evaluated or processed output.
    """

    def wrapped(*args: object) -> object:
        """Evaluate and process the wrapped operation.

        Args:
            *args (Any): Variable positional arguments.

        Returns:
            object: The evaluated or processed output.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError("Cannot emit Pmap outside of a tracing context.")
        dummy_args = []
        for a in args:
            if isinstance(a, Tensor):
                new_shape = a.shape[1:] if len(a.shape) > 0 else ()
                proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
                dummy_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
            else:
                dummy_args.append(a)
        body_graph = _trace_function(func, tuple(dummy_args), "pmap_body")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Pmap",
            inputs=[typing.cast(typing.Any, a).data.id for a in args if isinstance(a, Tensor)],
            attributes={"axis_name": axis_name, "body": body_graph},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        arg = typing.cast(typing.Any, args[0])
        proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
        return Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))

    return wrapped


def stop_gradient_tracing(x: object) -> object:
    """Evaluate and process the stop gradient tracing operation.

    Args:
        x (object): Required parameter for x.

    Returns:
        object: The evaluated or processed output.
    """
    if not global_tracing_state.is_tracing:
        return x
    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.data.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
        return Tensor(proxy, TensorConfig(x.shape, x.dtype, x.device))
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        return ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype)
    return x


def assert_value_tracing(condition: object, message: str = "") -> None:
    """Evaluate and process the assert value tracing operation.

    Args:
        condition (object): Required parameter for condition.
        message (str): Required parameter for message.

    Returns:
        Any: The evaluated or processed output.
    """
    if not global_tracing_state.is_tracing:
        record_assertion(condition, message)
        return
    inp_id = condition.data.id if isinstance(condition, Tensor) else condition.id
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Assert",
        inputs=[inp_id],
        attributes={"message": message},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
