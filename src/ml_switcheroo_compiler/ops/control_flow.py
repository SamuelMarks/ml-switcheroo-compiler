"""Provides higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.core import IRBlock
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer


def _trace_function(func: Callable, args: tuple[Tensor, ...], name: str) -> IRBlock:
    """Traces a Python function's execution into an IRBlock.

    This helper function temporarily redirects the active tracer to capture the
    operations
    performed by the given function when applied to proxy tensor arguments. It
    constructs
    input and output logical nodes to form a complete subgraph

    Args:
        func (Callable): The Python function to trace
        args (tuple[Tensor, ...]): The concrete or proxy tensor arguments to pass to the
        function
        name (str): The name to assign to the traced subgraph/IRBlock

    Returns:
    IRBlock: The traced intermediate representation block containing the logical
    nodes

    Raises:
    TypeError: If the traced function does not return a Tensor or a tuple of
    Tensors
    """
    prev_graph = _tracer.active_graph
    is_tracing = _tracer.is_tracing

    subgraph = _tracer.start_tracing(name=name)

    # Re-wrap input args as proxy tensors in the subgraph context
    proxy_args = []
    input_ids = []
    for _i, arg in enumerate(args):
        in_id = str(uuid.uuid4())
        node = LogicalNode(
            id=in_id,
            op_type="Input",
            inputs=[],
            shape_metadata=arg.shape,
        )
        subgraph.nodes[in_id] = node
        input_ids.append(in_id)
        proxy = ProxyTensor(id=in_id, shape=arg.shape, dtype=arg.dtype.value)
        proxy_tensor = Tensor(
            data=proxy,
            shape=arg.shape,
            dtype=arg.dtype,
            device=arg.device,
        )
        proxy_args.append(proxy_tensor)

    # Execute the function
    out = func(*proxy_args)

    # Process outputs
    if isinstance(out, Tensor):
        out_ids = [out.data.id]
    elif isinstance(out, (tuple, list)):
        out_ids = [o.data.id for o in out]
    else:
        msg = "Control flow functions must return a Tensor or a tuple of Tensors."
        raise TypeError(
            msg,
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

    return IRBlock(
        id=name,
        nodes=list(subgraph.nodes.values()),
        inputs=input_ids,
        outputs=[out_node.id],
    )


def cond(
    pred: Tensor,
    true_fn: Callable[[], Any],
    false_fn: Callable[[], Any],
) -> object:
    """Conditionally executes one of two functions based on a boolean predicate.

    In eager mode, this directly evaluates the predicate and runs the corresponding
    branch function. In tracing mode, it traces both branches into subgraphs and
    emits an 'If' logical node

    Args:
        pred (Tensor): A scalar boolean tensor determining which branch to execute
        true_fn (Callable[[], Any]): The function to execute if the predicate is True
        false_fn (Callable[[], Any]): The function to execute if the predicate is False

    Returns:
    object: The result of executing the selected branch (a Tensor or structure of
    Tensors)

    Raises:
    RuntimeError: If called outside of a tracing context when eager mode is
    disabled
    """
    if config.eager_mode:
        if bool(pred.data):
            return true_fn()
        return false_fn()
    if not _tracer.is_tracing:
        msg = "Cannot emit Cond node outside of a tracing context."
        raise RuntimeError(msg)

    true_graph = _trace_function(true_fn, (), "true_branch")
    false_graph = _trace_function(false_fn, (), "false_branch")

    # For tracing shapes, execute true_fn once eagerly? No, trace function provides
    # it
    # We need the output shapes
    # Actually in JAX/Switcheroo, output shapes/dtypes must match
    # We get output nodes from true_graph
    # We just approximate shape here
    # In a real compiler, we'd extract it from the graphs
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

    # We return a dummy proxy. In full implementation, we'd parse output shapes
    proxy = ProxyTensor(id=out_id, shape=(), dtype="float32")
    return Tensor(data=proxy, shape=(), dtype=DType.Float32, device=pred.device)


def while_loop(
    cond_fn: Callable[[Any], Tensor],
    body_fn: Callable[[Any], Any],
    init_val: object,
) -> object:
    """Executes a body function repeatedly while a condition function evaluates to True.

    In eager mode, this runs a standard Python while loop. In tracing mode, it
    traces
    both the condition and body functions into subgraphs and emits a 'Loop' logical
    node

    Args:
        cond_fn (Callable[[Any], Tensor]): A function that takes the current state and
        returns a scalar boolean Tensor
        body_fn (Callable[[Any], Any]): A function that takes the current state and
        returns the updated state of the same structure and shape
        init_val (object): The initial state, which can be a Tensor or a sequence of
        Tensors

    Returns:
    object: The final state after the loop terminates

    Raises:
    RuntimeError: If called outside of a tracing context when eager mode is
    disabled
    """
    if config.eager_mode:
        val = init_val
        while bool(cond_fn(val).data):
            val = body_fn(val)
        return val
    if not _tracer.is_tracing:
        msg = "Cannot emit While node outside of a tracing context."
        raise RuntimeError(msg)

    # Wrap state in a tuple for tracing
    args = (init_val,) if isinstance(init_val, Tensor) else tuple(init_val)

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
            id=out_id,
            shape=init_val.shape,
            dtype=init_val.dtype.value,
        )
        return Tensor(
            data=proxy,
            shape=init_val.shape,
            dtype=init_val.dtype,
            device=init_val.device,
        )
    # Tuple of tensors
    return init_val  # Simplified


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]],
    init: object,
    xs: object,
    length: int | None = None,
) -> tuple[Any, Any]:
    """Scans a function over the leading dimension of sequence tensors while carrying.

    state

    In eager mode, this sequentially applies the function along the 0-th dimension
    of
    the input sequence, accumulating the results. In tracing mode, it emits a 'Scan'
    logical node

    Args:
        f (Callable[[Any, Any], tuple[Any, Any]]): A function mapping `(carry, x)` to
        `(new_carry, y)`
        init (object): The initial carry state
        xs (object): The sequence of values to scan over, typically a Tensor
        length (int | None): An optional length to scan

    Returns:
    tuple[Any, Any]: A tuple containing the final carry state and the stacked
        outputs (ys) from each step

    Raises:
    RuntimeError: If called outside of a tracing context when eager mode is
    disabled
    """
    if config.eager_mode:
        return _scan_eager(f, init, xs, length)

    return _scan_tracing(init, xs)


def _scan_eager(f: Callable, init: object, xs: object, length: int | None) -> tuple[object, object]:
    """Execute _scan_eager.

    Args:
        f (Any): Argument f.
        init (Any): Argument init.
        xs (Any): Argument xs.
        length (Any): Argument length.

    Returns:
    Any: The result.
    """
    carry = init
    ys = []
    # xs is assumed to be a Tensor with a batch dimension 0
    scan_length = length if length is not None else (xs.shape[0] if xs is not None else 0)
    for i in range(scan_length):
        # Extract slice
        x = Tensor(xs.data[i], xs.shape[1:], xs.dtype, xs.device) if xs is not None else None
        carry, y = f(carry, x)
        ys.append(y.data if hasattr(y, "data") else y)

    out_tensor = _stack_scan_outputs(ys, init, y if scan_length > 0 else init)
    return carry, out_tensor


def _stack_scan_outputs(ys: list, init: object, last_y: object) -> Tensor:
    """Execute _stack_scan_outputs.

    Args:
        ys (Any): Argument ys.
        init (Any): Argument init.
        last_y (Any): Argument last_y.

    Returns:
    Any: The result.
    """
    if len(ys) > 0 and isinstance(ys[0], tuple):
        stacked_ys = get_active_backend().execute_op("Stack", ys)
        return Tensor(
            stacked_ys,
            stacked_ys.shape,
            last_y.dtype if hasattr(last_y, "dtype") else init.dtype,
            last_y.device if hasattr(last_y, "device") else init.device,
        )
    else:
        stacked_ys = get_active_backend().array(ys)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(
            stacked_ys,
            stacked_ys.shape,
            DType(str(stacked_ys.dtype)),
            config.default_device,
        )


def _scan_tracing(init: object, xs: object) -> tuple[object, object]:
    """Execute _scan_tracing.

    Args:
        init (Any): Argument init.
        xs (Any): Argument xs.

    Returns:
    Any: The result.
    """
    if not _tracer.is_tracing:
        msg = "Cannot emit Scan node outside of a tracing context."
        raise RuntimeError(msg)

    # Approximate tracing
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Scan",
        inputs=[init.data.id, xs.data.id] if isinstance(init, Tensor) else [xs.data.id],
        shape_metadata=(),
    )
    _tracer.add_node(node)

    # Return a dummy proxy
    proxy = ProxyTensor(id=out_id, shape=xs.shape, dtype=xs.dtype.value)
    out_tensor = Tensor(
        data=proxy,
        shape=xs.shape,
        dtype=xs.dtype,
        device=xs.device,
    )
    return init, out_tensor


def vmap(
    func: Callable,
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
) -> Callable:
    """Creates a vectorized version of a function mapped over specified axes.

    In eager mode, the returned function applies the original function sequentially
    over the batch dimension using a loop. In tracing mode, it records a 'Vmap'
    logical node in the IR

    Args:
        func (Callable): The function to vectorize
        in_axes (int | tuple[int, ...]): Specifies which axes of the inputs to map over
        Defaults to 0
        out_axes (int | tuple[int, ...]): Specifies where the mapped axis should appear
        in the outputs. Defaults to 0

    Returns:
    Callable: A vectorized version of the input function
    """

    def wrapped(*args: object) -> object:
        """Wrapped.

        Args:
            *args (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
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
                axes = in_axes if isinstance(in_axes, int) else in_axes[0]
                sliced_data = get_active_backend().execute_op("Take", arg.data, i, axis=axes)
                sliced_shape = tuple(s for j, s in enumerate(arg.shape) if j != axes)
                sliced_arg = Tensor(
                    sliced_data,
                    sliced_shape,
                    arg.dtype,
                    arg.device,
                )
                outs.append(func(sliced_arg).data)

            out_data = get_active_backend().execute_op(
                "Stack",
                outs,
                axis=out_axes if isinstance(out_axes, int) else out_axes[0],
            )
            out_shape = out_data.shape
            return Tensor(out_data, out_shape, arg.dtype, arg.device)
        if not _tracer.is_tracing:
            msg = "Cannot emit Vmap outside of a tracing context."
            raise RuntimeError(msg)

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
            data=proxy,
            shape=arg.shape,
            dtype=arg.dtype,
            device=arg.device,
        )

    return wrapped


def pmap(func: Callable, axis_name: str | None = None) -> Callable:
    """Creates a parallel mapped version of a function for distributed execution.

    In eager mode, this falls back to a vectorized map (vmap). In tracing mode,
    it records a 'Pmap' logical node in the IR

    Args:
        func (Callable): The function to map in parallel
        axis_name (str | None): The name of the mapped axis, used for collective
        operations within the function. Defaults to None

    Returns:
    Callable: A parallel mapped version of the input function
    """

    def wrapped(*args: object) -> object:
        """Wrapped.

        Args:
            *args (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if config.eager_mode:
            # In eager mode, pmap usually falls back to vmap or a loop
            return vmap(func)(*args)
        if not _tracer.is_tracing:
            msg = "Cannot emit Pmap outside of a tracing context."
            raise RuntimeError(msg)

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
            data=proxy,
            shape=arg.shape,
            dtype=arg.dtype,
            device=arg.device,
        )

    return wrapped


def stop_gradient(x: object) -> object:
    """Stops the flow of gradients during reverse-mode differentiation.

    Args:
        x (object): The x.

    Returns:
        object: The computed result.
    """
    import uuid

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    if config.eager_mode or not _tracer.is_tracing:
        return x

    if isinstance(x, Tensor) and isinstance(x.data, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="StopGradient",
            inputs=[x.data.id],
            shape_metadata=x.shape,
        )
        _tracer.add_node(node)
        proxy = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
        return Tensor(data=proxy, shape=x.shape, dtype=x.dtype, device=x.device)
    if isinstance(x, ProxyTensor):
        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="StopGradient",
            inputs=[x.id],
            shape_metadata=x.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype)

    return x
