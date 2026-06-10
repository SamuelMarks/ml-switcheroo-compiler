"""Higher-Order Control Flow Primitives."""

from typing import Callable, Any, Union
import uuid
import numpy as np

from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode, LogicalGraph


def _trace_function(
    func: Callable, args: tuple[Tensor, ...], name: str
) -> LogicalGraph:
    """Trace a Python function into a LogicalGraph."""
    prev_graph = _tracer.active_graph
    is_tracing = _tracer.is_tracing

    subgraph = _tracer.start_tracing(name=name)

    # Re-wrap input args as proxy tensors in the subgraph context
    proxy_args = []
    for _i, arg in enumerate(args):
        in_id = str(uuid.uuid4())
        node = LogicalNode(
            id=in_id,
            op_type="Input",
            inputs=[],
            shape_metadata=arg.shape,
        )
        subgraph.nodes[in_id] = node
        proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
        proxy_tensor = Tensor(
            data=proxy, shape=arg.shape, dtype=arg.dtype, device=arg.device
        )
        proxy_args.append(proxy_tensor)

    # Execute the function
    out = func(*proxy_args)

    # Process outputs
    if isinstance(out, Tensor):
        out_ids = [out.data.id]
    elif isinstance(out, tuple):
        out_ids = [o.data.id for o in out]
    else:
        raise TypeError(
            "Control flow functions must return a Tensor or a tuple of Tensors."
        )

    out_node = LogicalNode(
        id=str(uuid.uuid4()),
        op_type="Output",
        inputs=out_ids,
        shape_metadata=(),
    )
    subgraph.nodes[out_node.id] = out_node

    _tracer.stop_tracing()
    _tracer.active_graph = prev_graph
    _tracer.is_tracing = is_tracing

    return subgraph


def cond(pred: Tensor, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> Any:
    """Conditionally execute `true_fn` or `false_fn` based on `pred`.

    Args:
        pred: A scalar boolean Tensor.
        true_fn: Function to execute if pred is true.
        false_fn: Function to execute if pred is false.

    Returns:
        The output of the executed function.
    """
    if config.eager_mode:
        if bool(pred.data):
            return true_fn()
        else:
            return false_fn()
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Cond node outside of a tracing context.")

        true_graph = _trace_function(true_fn, (), "true_branch")
        false_graph = _trace_function(false_fn, (), "false_branch")

        # For tracing shapes, execute true_fn once eagerly? No, trace function provides it.  # noqa: E501
        # We need the output shapes.
        # Actually in JAX/Switcheroo, output shapes/dtypes must match.
        # We get output nodes from true_graph.
        # We just approximate shape here.
        # In a real compiler, we'd extract it from the graphs.
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="If",
            inputs=[pred.data.id],
            attributes={
                "then_branch": true_graph,
                "else_branch": false_graph,
            },
            shape_metadata=(),  # Deferred shape resolution
        )
        _tracer.add_node(node)

        # We return a dummy proxy. In full implementation, we'd parse output shapes.
        proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")
        return Tensor(data=proxy, shape=(), dtype=DType.Float32, device=pred.device)


def while_loop(
    cond_fn: Callable[[Any], Tensor], body_fn: Callable[[Any], Any], init_val: Any
) -> Any:
    """Execute `body_fn` while `cond_fn` is true.

    Args:
        cond_fn: Function taking state and returning a scalar boolean Tensor.
        body_fn: Function taking state and returning updated state.
        init_val: Initial state.

    Returns:
        The final state.
    """
    if config.eager_mode:
        val = init_val
        while bool(cond_fn(val).data):
            val = body_fn(val)
        return val
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit While node outside of a tracing context.")

        # Wrap state in a tuple for tracing
        if isinstance(init_val, Tensor):
            args = (init_val,)
        else:
            args = tuple(init_val)

        cond_graph = _trace_function(cond_fn, args, "cond")
        body_graph = _trace_function(body_fn, args, "body")

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Loop",
            inputs=[a.data.id for a in args],
            attributes={
                "body": body_graph,
                "cond": cond_graph,
            },
            shape_metadata=(),
        )
        _tracer.add_node(node)

        # Approximate return type
        if isinstance(init_val, Tensor):
            proxy = ProxyTensor(
                id=out_id, shape=init_val.shape, dtype=init_val.dtype.value
            )
            return Tensor(
                data=proxy,
                shape=init_val.shape,
                dtype=init_val.dtype,
                device=init_val.device,
            )
        else:
            # Tuple of tensors
            return init_val  # Simplified


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]], init: Any, xs: Any
) -> tuple[Any, Any]:
    """Scan a function over a sequence.

    Args:
        f: Function `(carry, x) -> (carry, y)`.
        init: Initial carry value.
        xs: Sequence of values to scan over.

    Returns:
        Tuple of `(final_carry, stacked_ys)`.
    """
    if config.eager_mode:
        carry = init
        ys = []
        # xs is assumed to be a Tensor with a batch dimension 0
        length = xs.shape[0]
        for i in range(length):
            # Extract slice
            x = Tensor(xs.data[i], xs.shape[1:], xs.dtype, xs.device)
            carry, y = f(carry, x)
            ys.append(y.data)
        stacked_ys = np.stack(ys)
        return carry, Tensor(
            stacked_ys,
            (length,) + ys[0].shape,
            ys[0].dtype if isinstance(ys[0], Tensor) else DType.Float32,
            xs.device,
        )
    else:
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot emit Scan node outside of a tracing context.")

        # Approximate tracing
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Scan",
            inputs=[init.data.id, xs.data.id]
            if isinstance(init, Tensor)
            else [xs.data.id],
            shape_metadata=(),
        )
        _tracer.add_node(node)

        # Return a dummy proxy
        proxy = ProxyTensor(id=out_id, shape=xs.shape, dtype=xs.dtype.value)
        out_tensor = Tensor(
            data=proxy, shape=xs.shape, dtype=xs.dtype, device=xs.device
        )
        return init, out_tensor


def vmap(
    func: Callable,
    in_axes: Union[int, tuple[int, ...]] = 0,
    out_axes: Union[int, tuple[int, ...]] = 0,
) -> Callable:
    """Vectorizing map.

    Args:
        func: The function to map.
        in_axes: Specifies which axes to map over for inputs.
        out_axes: Specifies where the mapped axis should appear in outputs.

    Returns:
        Vectorized version of `func`.
    """

    def wrapped(*args: Any) -> Any:
        if config.eager_mode:
            # Eager vmap uses a Python loop over the batch dimension
            # Simplified implementation for single tensor
            arg = args[0]
            batch_size = (
                (arg.shape[in_axes] if arg.shape else 1)
                if isinstance(in_axes, int)
                else (arg.shape[in_axes[0]] if arg.shape else 1)
            )
            outs = []
            for i in range(batch_size):
                if isinstance(in_axes, int):
                    sliced_data = np.take(arg.data, i, axis=in_axes)
                    sliced_shape = tuple(
                        s for j, s in enumerate(arg.shape) if j != in_axes
                    )
                    sliced_arg = Tensor(
                        sliced_data, sliced_shape, arg.dtype, arg.device
                    )
                    outs.append(func(sliced_arg).data)

            out_data = np.stack(
                outs, axis=out_axes if isinstance(out_axes, int) else out_axes[0]
            )
            out_shape = out_data.shape
            return Tensor(out_data, out_shape, arg.dtype, arg.device)
        else:
            if not _tracer.is_tracing:
                raise RuntimeError("Cannot emit Vmap outside of a tracing context.")

            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Vmap",
                inputs=[a.data.id for a in args if isinstance(a, Tensor)],
                attributes={"in_axes": in_axes, "out_axes": out_axes},
                shape_metadata=(),
            )
            _tracer.add_node(node)
            # Dummy return
            arg = args[0]
            proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
            return Tensor(
                data=proxy, shape=arg.shape, dtype=arg.dtype, device=arg.device
            )

    return wrapped


def pmap(func: Callable, axis_name: str = None) -> Callable:
    """Parallel map for distributed execution.

    Args:
        func: The function to map.
        axis_name: Name of the mapped axis for collective operations.

    Returns:
        Parallel mapped version of `func`.
    """

    def wrapped(*args: Any) -> Any:
        if config.eager_mode:
            # In eager mode, pmap usually falls back to vmap or a loop
            return vmap(func)(*args)
        else:
            if not _tracer.is_tracing:
                raise RuntimeError("Cannot emit Pmap outside of a tracing context.")

            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Pmap",
                inputs=[a.data.id for a in args if isinstance(a, Tensor)],
                attributes={"axis_name": axis_name},
                shape_metadata=(),
            )
            _tracer.add_node(node)
            # Dummy return
            arg = args[0]
            proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
            return Tensor(
                data=proxy, shape=arg.shape, dtype=arg.dtype, device=arg.device
            )

    return wrapped
