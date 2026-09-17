"""Module vmap.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Provide higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation.
"""


import uuid
from typing import Any, Callable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def _eager_vmap(
    func,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args,
):
    """Evaluate _eager_vmap operation using vectorized IR execution.

    Args:
        func (Callable): The function to vectorize.
        in_axes (int | tuple[int, ...]): Axis or axes mapped over in inputs.
        out_axes (int | tuple[int, ...]): Axis or axes in output tensor(s).
        args (tuple): Input arguments.

    Returns:
        Tensor: Vectorized output tensor.
    """
    first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
    if first_tensor is None:
        return func(*args)

    in_axis = in_axes if isinstance(in_axes, int) else (in_axes[0] if in_axes else 0)
    batch_size = first_tensor.shape[in_axis] if first_tensor.shape and in_axis < len(first_tensor.shape) else 1

    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_block = _trace_function(func, tuple(symbolic_args), "vmap_body")

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorize_graph

    body_graph = IRGraph(name="vmap_body")
    body_graph.inputs = list(body_block.inputs)
    body_graph.outputs = list(body_block.outputs)
    nodes_iter = body_block.nodes.values() if isinstance(body_block.nodes, dict) else body_block.nodes
    for n in nodes_iter:
        body_graph.nodes[n.id] = IRNode(**n.__dict__) if not isinstance(n, IRNode) else n

    vectorized_graph = vectorize_graph(body_graph, in_axes=in_axes, batch_size=batch_size, out_axes=out_axes)

    backend = get_active_backend()
    input_map = {}
    for inp_id, conc_arg in zip(body_block.inputs, args):
        val = conc_arg.data if isinstance(conc_arg, Tensor) else conc_arg
        input_map[inp_id] = val

    out_map = evaluate_graph(vectorized_graph, input_map, backend=backend)
    out_node_id = vectorized_graph.outputs[0]
    out_val = out_map[out_node_id]

    if hasattr(backend, "asarray"):
        out_data = backend.asarray(out_val)
    else:
        out_data = out_val
    out_shape = tuple(out_data.shape) if hasattr(out_data, "shape") else (batch_size,)
    return Tensor(out_data, TensorConfig(out_shape, first_tensor.dtype, first_tensor.device))


def _resolve_vmap_axis(in_axes: int | tuple[int, ...], i: int) -> int | None:
    """Evaluate _resolve_vmap_axis operation.

    Args:
        in_axes (Any): The in_axes parameter.
        i (int): The i parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return in_axes if isinstance(in_axes, int) else (in_axes[i] if i < len(in_axes) else 0)


def _compute_vmap_shape(a: Tensor, axis: int | None):
    """Evaluate _compute_vmap_shape operation.

    Args:
        a (Tensor): The a parameter.
        axis (Any): The axis parameter.

    Returns:
        tuple: Result.
    """
    if axis is not None and len(a.shape) > 0:
        return tuple(s for j, s in enumerate(a.shape) if j != axis)
    return a.shape


def _create_vmap_symbolic_args(args, in_axes: int | tuple[int, ...]):
    """Evaluate _create_vmap_symbolic_args operation.

    Args:
        args (Any): The args parameter.
        in_axes (Any): The in_axes parameter.

    Returns:
            tuple[int, ...]: Result.
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
    func,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args,
):
    """Evaluate _trace_vmap operation.

    Args:
        func (Callable): The func parameter.
        in_axes (Any): The in_axes parameter.
        out_axes (Any): The out_axes parameter.
        args (tuple): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_graph = _trace_function(func, tuple(symbolic_args), "vmap_body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Vmap",
        inputs=[a.data.id for a in args if isinstance(a, Tensor)],
        attributes={"in_axes": in_axes, "out_axes": out_axes, "body": body_graph},
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    arg = args[0]
    proxy = ProxyTensor(id=out_id, shape=arg.shape, dtype=arg.dtype.value)
    return Tensor(proxy, TensorConfig(arg.shape, arg.dtype, arg.device))


def vmap(
    func,
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
):
    """Create a vectorized version of a function mapped over specified axes.

    Args:
        func (Callable): The func parameter.
        in_axes (Any): The in_axes parameter.
        out_axes (Any): The out_axes parameter.

    Returns:
        Callable: Result.
    """

    def wrapped(*args):
        """Wrap.

        Args:
            *args (Any): Positional args.

        Returns:
            tuple[int, ...]: Result.
        """
        if config.eager_mode:
            return _eager_vmap(func, in_axes, out_axes, args)
        if not global_tracing_state.is_tracing:
            msg = "Cannot emit Vmap outside of a tracing context."
            raise RuntimeError(msg)
        return _trace_vmap(func, in_axes, out_axes, args)

    return wrapped
