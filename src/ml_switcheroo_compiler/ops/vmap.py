"""Provide higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation.
"""

from __future__ import annotations

import typing
import uuid
from collections.abc import Iterable

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import EagerCallableProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def _eager_vmap(
    func: TensorOrScalar,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: TensorOrScalar,
) -> TensorOrScalar:
    """Evaluate _eager_vmap operation using vectorized IR execution.

    Args:
        func (TensorOrScalar): The function to vectorize.
        in_axes (int | tuple[int, ...]): Axis or axes mapped over in inputs.
        out_axes (int | tuple[int, ...]): Axis or axes in output tensor(s).
        args (tuple): Input arguments.

    Returns:
        Tensor: Vectorized output tensor.
    """
    first_tensor = next((a for a in getattr(args, "__iter__")() if isinstance(a, Tensor)), None)
    if first_tensor is None:
        return typing.cast(TypedVariadicCallable[TensorOrScalar], func)(*getattr(args, "__iter__")())

    in_axis = in_axes if isinstance(in_axes, int) else (in_axes[0] if in_axes else 0)
    batch_size = first_tensor.shape[in_axis] if first_tensor.shape and in_axis < len(first_tensor.shape) else 1

    from ml_switcheroo_ir import LogicalGraph

    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_block = typing.cast("TypedVariadicCallable[LogicalGraph]", _trace_function)(func, tuple(getattr(symbolic_args, "__iter__")()), "vmap_body")

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorize_graph

    body_graph = IRGraph(name="vmap_body")
    body_graph.inputs = list(body_block.inputs)
    body_graph.outputs = list(body_block.outputs)
    nodes_iter: Iterable[LogicalNode] = body_block.nodes.values() if hasattr(body_block.nodes, "values") else typing.cast(Iterable[LogicalNode], body_block.nodes)
    for n in nodes_iter:
        body_graph.nodes[n.id] = IRNode(**n.__dict__) if type(n) is not IRNode else n

    vectorized_graph = vectorize_graph(body_graph, in_axes=in_axes, batch_size=batch_size, out_axes=out_axes)

    backend = get_active_backend()
    input_map = {}
    for inp_id, conc_arg in zip(body_block.inputs, getattr(args, "__iter__")()):
        val = conc_arg.data if isinstance(conc_arg, Tensor) else conc_arg
        input_map[inp_id] = val

    out_map = evaluate_graph(vectorized_graph, typing.cast("dict[str, EvalValue]", input_map), backend=backend)
    out_node_id = vectorized_graph.outputs[0]
    out_val = out_map[out_node_id]

    if hasattr(backend, "asarray"):
        out_data = backend.asarray(out_val)
    else:
        out_data = out_val
    out_shape = tuple(getattr(out_data, "shape")) if hasattr(out_data, "shape") else (batch_size,)
    return Tensor(out_data, TensorConfig(out_shape, first_tensor.dtype, first_tensor.device))


def _resolve_vmap_axis(in_axes: int | tuple[int, ...], i: int) -> int | None:
    """Compute the resolve vmap axis tensor operation.

    Args:
        in_axes (int | tuple[int, ...]): The in_axes parameter.
        i (int): The i parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return in_axes if isinstance(in_axes, int) else (in_axes[i] if i < len(in_axes) else 0)


def _compute_vmap_shape(a: Tensor, axis: int | None) -> tuple[int, ...]:
    """Compute the compute vmap shape tensor operation.

    Args:
        a (Tensor): The a parameter.
        axis (int | None): The axis parameter.

    Returns:
        tuple: Result.
    """
    if axis is not None and len(a.shape) > 0:
        return tuple(s for j, s in enumerate(a.shape) if j != axis)
    return a.shape


def _create_vmap_symbolic_args(args: TensorOrScalar, in_axes: int | tuple[int, ...]) -> TensorOrScalar:
    """Compute the create vmap symbolic args tensor operation.

    Args:
        args (TensorOrScalar): The args parameter.
        in_axes (int | tuple[int, ...]): The in_axes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    symbolic_args = []
    for i, a in enumerate(getattr(args, "__iter__")()):
        if isinstance(a, Tensor):
            axis = _resolve_vmap_axis(in_axes, i)
            new_shape = _compute_vmap_shape(a, axis)
            proxy = typing.cast("EagerCallableProtocol", ProxyTensor)(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
            symbolic_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
        else:
            symbolic_args.append(a)
    return symbolic_args


def _trace_vmap(
    func: TensorOrScalar,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: TensorOrScalar,
) -> TensorOrScalar:
    """Compute the trace vmap tensor operation.

    Args:
        func (TensorOrScalar): The func parameter.
        in_axes (int | tuple[int, ...]): The in_axes parameter.
        out_axes (int | tuple[int, ...]): The out_axes parameter.
        args (tuple): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from collections.abc import Mapping

    from ml_switcheroo_ir import AttributeValue, LogicalGraph

    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_graph = typing.cast("TypedVariadicCallable[LogicalGraph]", _trace_function)(func, tuple(getattr(symbolic_args, "__iter__")()), "vmap_body")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="Vmap",
        inputs=[getattr(a.data, "id") for a in getattr(args, "__iter__")() if isinstance(a, Tensor)],
        attributes=typing.cast("Mapping[str, AttributeValue]", {"in_axes": in_axes, "out_axes": out_axes, "body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    arg = getattr(args, "__getitem__")(0)
    proxy = ProxyTensor(id=out_id, shape=getattr(arg, "shape"), dtype=getattr(getattr(arg, "dtype"), "value"))
    return Tensor(proxy, TensorConfig(getattr(arg, "shape"), getattr(arg, "dtype"), getattr(arg, "device")))


def vmap(
    func: TensorOrScalar,
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
) -> TensorOrScalar:
    """Create a vectorized version of a function mapped over specified axes.

    Args:
        func (Callable): The func parameter.
        in_axes (int | tuple[int, ...]): The in_axes parameter.
        out_axes (int | tuple[int, ...]): The out_axes parameter.

    Returns:
        Callable: Result.
    """

    def wrapped(*args: TensorOrScalar) -> TensorOrScalar:
        """Wrap.

        Args:
            *args (TensorOrScalar): Positional args.

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
