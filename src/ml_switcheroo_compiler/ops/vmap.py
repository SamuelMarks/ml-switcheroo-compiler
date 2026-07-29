"""Provides higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation
"""

from __future__ import annotations

import uuid
from typing import Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def _eager_vmap(
    func: Callable,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: tuple[object, ...],
) -> object:
    """Evaluate and process the eager vmap operation.

    Args:
        func (Callable): Required parameter for func.
        in_axes (Any): Required parameter for in_axes.
        out_axes (Any): Required parameter for out_axes.
        args (tuple): Required parameter for args.

    Returns:
        object: The evaluated or processed output.
    """
    arg = args[0]
    in_axis = in_axes if isinstance(in_axes, int) else in_axes[0]
    out_axis = out_axes if isinstance(out_axes, int) else out_axes[0]

    batch_size = arg.shape[in_axis] if arg.shape else 1

    outs = []
    backend = get_active_backend()
    for i in range(batch_size):
        sliced_data = backend.execute_op("Take", arg.data, i, axis=in_axis)
        sliced_shape = tuple(s for j, s in enumerate(arg.shape) if j != in_axis)
        sliced_arg = Tensor(sliced_data, TensorConfig(sliced_shape, arg.dtype, arg.device))
        outs.append(func(sliced_arg).data)

    out_data = backend.execute_op("Stack", outs, axis=out_axis)
    return Tensor(out_data, TensorConfig(out_data.shape, arg.dtype, arg.device))


def _resolve_vmap_axis(in_axes: int | tuple[int, ...], i: int) -> int | None:
    """Evaluate and process the resolve vmap axis operation.

    Args:
        in_axes (Any): Required parameter for in_axes.
        i (int): Required parameter for i.

    Returns:
        Any: The evaluated or processed output.
    """
    return in_axes if isinstance(in_axes, int) else (in_axes[i] if i < len(in_axes) else 0)


def _compute_vmap_shape(a: Tensor, axis: int | None) -> tuple[int, ...]:
    """Evaluate and process the compute vmap shape operation.

    Args:
        a (Tensor): Required parameter for a.
        axis (Any): Required parameter for axis.

    Returns:
        tuple: The evaluated or processed output.
    """
    if axis is not None and len(a.shape) > 0:
        return tuple(s for j, s in enumerate(a.shape) if j != axis)
    return a.shape


def _create_vmap_symbolic_args(args: tuple[object, ...], in_axes: int | tuple[int, ...]) -> list[object]:
    """Evaluate and process the create vmap dummy args operation.

    Args:
        args (tuple): Required parameter for args.
        in_axes (Any): Required parameter for in_axes.

    Returns:
        list: The evaluated or processed output.
    """
    symbolic_args = []
    for i, a in enumerate(args):
        if isinstance(a, Tensor):
            axis = _resolve_vmap_axis(in_axes, i)
            new_shape = _compute_vmap_shape(a, axis)
            proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
            symbolic_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
        else:
            symbolic_args.append(a)
    return symbolic_args


def _trace_vmap(
    func: Callable,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: tuple[object, ...],
) -> object:
    """Evaluate and process the trace vmap operation.

    Args:
        func (Callable): Required parameter for func.
        in_axes (Any): Required parameter for in_axes.
        out_axes (Any): Required parameter for out_axes.
        args (tuple): Required parameter for args.

    Returns:
        object: The evaluated or processed output.
    """
    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_graph = _trace_function(func, tuple(symbolic_args), "vmap_body")

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Vmap",
        inputs=[a.data.id for a in args if isinstance(a, Tensor)],
        attributes={"in_axes": in_axes, "out_axes": out_axes, "body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)

    arg = args[0]
    proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
    return Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))


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
            return _eager_vmap(func, in_axes, out_axes, args)

        if not global_tracing_state.is_tracing:
            msg = "Cannot emit Vmap outside of a tracing context."
            raise RuntimeError(msg)

        return _trace_vmap(func, in_axes, out_axes, args)

    return wrapped
