"""Provide higher-order control flow primitives for tracing and eager execution.

This module implements functional control flow operators such as conditional branching,
while loops, scanning, vectorizing maps (vmap), and parallel maps (pmap). These
operators support both eager execution (using NumPy/Python loops) and tracing into an
intermediate representation (IR) graph for compilation.
"""

from __future__ import annotations

import typing
import uuid
from collections.abc import Iterable, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import ensure_callable, ensure_eager_list, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TensorOrScalar,
    TracedDataProtocol,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import as_node_list, clone_logical_node, ensure_attrs
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


@typing.runtime_checkable
class _NodesValuesProtocol(typing.Protocol):
    """Protocol for node mappings exposing a `.values()` method."""

    def values(self) -> Iterable[LogicalNode]:
        """Return node values.

        Returns:
            Iterable[LogicalNode]: Node values.
        """
        ...


@typing.runtime_checkable
class _AsArrayBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `.asarray(val)`."""

    def asarray(self, val: TensorOrScalar) -> TensorOrScalar:
        """Convert value to backend array.

        Args:
            val (TensorOrScalar): Input value.

        Returns:
            TensorOrScalar: Backend array.
        """
        ...


def _eager_vmap(
    func: TensorOrScalar,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: Sequence[TensorOrScalar],
) -> TensorOrScalar:
    """Evaluate _eager_vmap operation using vectorized IR execution.

    Args:
        func (TensorOrScalar): The function to vectorize.
        in_axes (int | tuple[int, ...]): Axis or axes mapped over in inputs.
        out_axes (int | tuple[int, ...]): Axis or axes in output tensor(s).
        args (Sequence[TensorOrScalar]): Input arguments.

    Returns:
        Tensor: Vectorized output tensor.
    """
    first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
    if first_tensor is None:
        return ensure_callable(func)(*args)

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
    raw_nodes: dict[str, LogicalNode] | list[LogicalNode] | _NodesValuesProtocol = body_block.nodes
    nodes_iter: Iterable[LogicalNode] = raw_nodes.values() if isinstance(raw_nodes, _NodesValuesProtocol) else as_node_list(body_block.nodes)
    for n in nodes_iter:
        body_graph.nodes[n.id] = clone_logical_node(n) if type(n) is not IRNode else n

    vectorized_graph = vectorize_graph(body_graph, in_axes=in_axes, batch_size=batch_size, out_axes=out_axes)

    backend = get_active_backend()
    input_map = {}
    for inp_id, conc_arg in zip(body_block.inputs, args):
        val = conc_arg.data if isinstance(conc_arg, Tensor) else conc_arg
        input_map[inp_id] = val

    out_map = evaluate_graph(vectorized_graph, input_map, backend=backend)
    out_node_id = vectorized_graph.outputs[0]
    out_val = out_map[out_node_id]

    raw_backend: TensorOrScalar | _AsArrayBackendProtocol = backend
    if isinstance(raw_backend, _AsArrayBackendProtocol):
        out_data = raw_backend.asarray(out_val)
    else:
        out_data = out_val
    out_shape = extract_shape(out_data) if isinstance(out_data, ShapeProtocol) else (batch_size,)
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


def _create_vmap_symbolic_args(args: Sequence[TensorOrScalar] | TensorOrScalar, in_axes: int | tuple[int, ...]) -> list[TensorOrScalar]:
    """Compute the create vmap symbolic args tensor operation.

    Args:
        args (Sequence[TensorOrScalar] | TensorOrScalar): The args parameter.
        in_axes (int | tuple[int, ...]): The in_axes parameter.

    Returns:
        list[TensorOrScalar]: Result.
    """
    args_seq: Sequence[TensorOrScalar] = args if isinstance(args, (list, tuple)) else ensure_eager_list(args)
    symbolic_args: list[TensorOrScalar] = []
    for i, a in enumerate(args_seq):
        if isinstance(a, Tensor):
            axis = _resolve_vmap_axis(in_axes, i)
            new_shape = _compute_vmap_shape(a, axis)
            proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
            symbolic_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
        else:
            symbolic_args.append(a)
    return symbolic_args


def _as_traced_data(val: TensorOrScalar) -> TracedDataProtocol:
    """Narrow a traced tensor data field to `TracedDataProtocol`.

    Args:
        val (TensorOrScalar): Tensor data field.

    Returns:
        TracedDataProtocol: Narrowed traced data object.
    """

    def _guard(cand: TensorOrScalar) -> typing.TypeGuard[TracedDataProtocol]:
        """Type guard narrowing `cand` to `TracedDataProtocol`.

        Args:
            cand (TensorOrScalar): Candidate value.

        Returns:
            bool: True if non-None.
        """
        return cand is not None

    assert _guard(val)
    return val


def _trace_vmap(
    func: TensorOrScalar,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: Sequence[TensorOrScalar],
) -> TensorOrScalar:
    """Compute the trace vmap tensor operation.

    Args:
        func (TensorOrScalar): The func parameter.
        in_axes (int | tuple[int, ...]): The in_axes parameter.
        out_axes (int | tuple[int, ...]): The out_axes parameter.
        args (Sequence[TensorOrScalar]): The args parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_graph = _trace_function(func, tuple(symbolic_args), "vmap_body")
    out_id = str(uuid.uuid4())
    node_inputs: list[str] = [_as_traced_data(a.data).id for a in args if isinstance(a, Tensor)]
    node = LogicalNode(
        id=out_id,
        op_type="Vmap",
        inputs=node_inputs,
        attributes=ensure_attrs({"in_axes": in_axes, "out_axes": out_axes, "body": body_graph}),
        subgraphs={"body": body_graph},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)
    arg = args[0]
    arg_shape = extract_shape(arg)
    arg_dtype = extract_dtype(arg)
    dtype_str = arg_dtype.value if isinstance(arg_dtype, DType) else str(arg_dtype)
    proxy = ProxyTensor(id=out_id, shape=arg_shape, dtype=dtype_str)
    return Tensor(proxy, TensorConfig(arg_shape, arg_dtype, extract_device(arg)))


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
