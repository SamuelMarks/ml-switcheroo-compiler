"""Module tracing.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Tracing implementations for control flow operations."""


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


def cond_tracing(pred: Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> Any:  # type: ignore
    """Evaluate cond_tracing operation.

    Args:
        pred (Tensor): The pred parameter.
        true_fn (Callable): The true_fn parameter.
        false_fn (Callable): The false_fn parameter.

    Returns: Any: Result.
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
        inputs=[pred.data.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        attributes={"then_branch": true_graph, "else_branch": false_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")  # type: ignore
    return Tensor(proxy, TensorConfig((), DType.Float32, pred.device))


def while_loop_tracing(cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: Any) -> Any:  # type: ignore
    """Evaluate while_loop_tracing operation.

    Args:
        cond_fn (object): The cond_fn parameter.
        body_fn (object): The body_fn parameter.
        init_val (object): The init_val parameter.

    Returns: Any: Result.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit While node outside of a tracing context.")
    args: tuple[Any, ...] = (init_val,) if isinstance(init_val, Tensor) else tuple(init_val)
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
        proxy = ProxyTensor(id=out_id, shape=init_val.shape, dtype=init_val.dtype.value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        return Tensor(proxy, TensorConfig(init_val.shape, init_val.dtype, init_val.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return init_val


def _flatten_inputs(obj: Any) -> list[str]:
    """Evaluate _flatten_inputs operation.

    Args:
        obj (object): The obj parameter.

    Returns: Any: Result.
    """
    if isinstance(obj, Tensor):
        return [obj.data.id]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    elif isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            res.extend(_flatten_inputs(item))
        return res
    return []


def scan_tracing(f: Callable[..., Any], init: Any, xs: Any, length: int | None = None) -> tuple[Any, Any]:
    """Evaluate scan_tracing operation.

    Args:
        f (Callable): The f parameter.
        init (object): The init parameter.
        xs (object): The xs parameter.
        length (object): The length parameter.

    Returns:
        tuple: Result.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Scan node outside of a tracing context.")
    x_shape = xs.shape[1:] if xs is not None and len(xs.shape) > 0 else ()
    proxy_x = ProxyTensor(id="proxy_x_tensor", shape=x_shape, dtype=xs.dtype.value)  # type: ignore
    proxy_x_tensor = Tensor(proxy_x, TensorConfig(x_shape, xs.dtype, xs.device))
    body_graph = _trace_function(f, (init, proxy_x_tensor), "scan_body")
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
    proxy = ProxyTensor(id=out_id, shape=xs.shape, dtype=xs.dtype.value)  # type: ignore
    out_tensor = Tensor(proxy, TensorConfig(xs.shape, xs.dtype, xs.device))
    return init, out_tensor


def map_fn_tracing(fn: Callable[..., Any], elems: Tensor, dtype: DType | None = None) -> Any:  # type: ignore
    """Evaluate map_fn_tracing operation.

    Args:
        fn (Callable): The fn parameter.
        elems (Tensor): The elems parameter.
        dtype (object): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    if not global_tracing_state.is_tracing:
        from ml_switcheroo_compiler.core.errors import TracingError

        raise TracingError("Cannot emit Map node outside of a tracing context.")
    x_shape = elems.shape[1:] if elems is not None and len(elems.shape) > 0 else ()
    proxy_x = ProxyTensor(id="proxy_x_tensor", shape=x_shape, dtype=elems.dtype.value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    proxy_x_tensor = Tensor(proxy_x, TensorConfig(x_shape, elems.dtype, elems.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    body_graph = _trace_function(fn, (proxy_x_tensor,), "map_body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Map",
        inputs=[elems.data.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        attributes={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    out_dtype = dtype if dtype is not None else elems.dtype
    out_shape = (elems.shape[0],)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)  # type: ignore
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, elems.device))


def pmap_tracing(func: Callable[..., Any], axis_name: str | None = None) -> Callable[..., Any]:
    """Evaluate pmap_tracing operation.

    Args:
        func (Callable): The func parameter.
        axis_name (object): The axis_name parameter.

    Returns:
        Callable: Result.
    """

    def wrapped(*args: Any) -> Any:
        """Evaluate wrapped operation.

        Args:
            *args (object): Positional args.

        Returns: Any: Result.
        """
        if not global_tracing_state.is_tracing:
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError("Cannot emit Pmap outside of a tracing context.")
        proxy_args = []
        for a in args:
            if isinstance(a, Tensor):
                new_shape = a.shape[1:] if len(a.shape) > 0 else ()
                proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
                proxy_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            else:
                proxy_args.append(a)
        body_graph = _trace_function(func, tuple(proxy_args), "pmap_body")
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Pmap",
            inputs=[str(getattr(getattr(a, "data", None), "id", "")) for a in args if isinstance(a, Tensor)],
            attributes={"axis_name": axis_name, "body": body_graph},
            shape_metadata=(),
        )
        global_tracing_state.add_node(node)
        arg = args[0]
        proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
        return Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))

    return wrapped


def stop_gradient_tracing(x: Any) -> Any:
    """Evaluate stop_gradient_tracing operation.

    Args:
        x (object): The x parameter.

    Returns: Any: Result.
    """
    if not global_tracing_state.is_tracing:
        return x
    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.data.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        return Tensor(proxy, TensorConfig(x.shape, x.dtype, x.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(id=out_id, op_type="StopGradient", inputs=[x.id], shape_metadata=x.shape)
        global_tracing_state.add_node(node)
        return ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype)
    return x


def assert_value_tracing(condition: Any, message: str = "") -> None:
    """Evaluate assert_value_tracing operation.

    Args:
        condition (object): The condition parameter.
        message (str): The message parameter.

    Returns:
        NoneType: Result.
    """
    if not global_tracing_state.is_tracing:
        record_assertion(condition, message)
        return
    inp_id = condition.data.id if isinstance(condition, Tensor) else condition.id  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Assert",
        inputs=[inp_id],
        attributes={"message": message},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
