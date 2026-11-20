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

from ml_switcheroo_compiler.backends.registry import coerce_operand_list, get_active_backend, require_callable
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeProtocol,
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

    def asarray(self, val: OpValue) -> OpValue:
        """Convert value to backend array.

        Args:
            val (OpValue): Input value.

        Returns:
            OpValue: Backend array.
        """
        ...


def _eager_vmap(
    func: OpValue,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: Sequence[OpValue],
) -> OpValue:
    """Evaluate _eager_vmap operation using vectorized IR execution.

    Args:
        func (OpValue): The function to vectorize.
        in_axes (int | tuple[int, ...]): Axis or axes mapped over in inputs.
        out_axes (int | tuple[int, ...]): Axis or axes in output tensor(s).
        args (Sequence[OpValue]): Input arguments.

    Returns:
        Tensor: Vectorized output tensor.
    """
    first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
    if first_tensor is None:
        return require_callable(func)(*args)

    in_axis = in_axes if isinstance(in_axes, int) else (in_axes[0] if in_axes else 0)
    ft_shape = extract_shape(first_tensor)
    batch_size = ft_shape[in_axis] if ft_shape and in_axis < len(ft_shape) else 1

    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_block = _trace_function(func, tuple(symbolic_args), "vmap_body")

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorize_graph

    body_graph = IRGraph(name="vmap_body")
    body_graph.inputs = list(body_block.inputs)
    body_graph.outputs = list(body_block.outputs)
    raw_nodes: dict[str, LogicalNode] | list[LogicalNode] | _NodesValuesProtocol = body_block.nodes
    nodes_iter: Iterable[LogicalNode] = (
        raw_nodes.values() if isinstance(raw_nodes, _NodesValuesProtocol) else as_node_list(body_block.nodes)
    )
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

    raw_backend: OpValue | _AsArrayBackendProtocol = backend
    if isinstance(raw_backend, _AsArrayBackendProtocol):
        out_data = raw_backend.asarray(out_val)
    else:
        out_data = out_val
    out_shape = extract_shape(out_data) if isinstance(out_data, ShapeProtocol) else (batch_size,)
    return Tensor(out_data, TensorConfig(out_shape, first_tensor.dtype, first_tensor.device))


def _resolve_vmap_axis(in_axes: int | tuple[int, ...], i: int) -> int | None:
    """Compute the resolve vmap axis tensor operation.

    Args:
        in_axes (int | tuple[int, ...]): Input `in_axes` argument (in axes).
        i (int): Input `i` argument (i).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return in_axes if isinstance(in_axes, int) else (in_axes[i] if i < len(in_axes) else 0)


def _compute_vmap_shape(a: Tensor, axis: int | None) -> tuple[int, ...]:
    """Compute the compute vmap shape tensor operation.

    Args:
        a (Tensor): Input `a` argument (a).
        axis (int | None): Input `axis` argument (axis).

    Returns:
        tuple: Computed `tuple` output.
    """
    a_shape = extract_shape(a)
    if axis is not None and len(a_shape) > 0:
        return tuple(s for j, s in enumerate(a_shape) if j != axis)
    return a_shape


def _create_vmap_symbolic_args(args: Sequence[OpValue] | OpValue, in_axes: int | tuple[int, ...]) -> list[OpValue]:
    """Compute the create vmap symbolic args tensor operation.

    Args:
        args (Sequence[OpValue] | OpValue): Input `args` argument (args).
        in_axes (int | tuple[int, ...]): Input `in_axes` argument (in axes).

    Returns:
        list[OpValue]: Computed `list[OpValue]` output.
    """
    args_seq: Sequence[OpValue] = args if isinstance(args, (list, tuple)) else coerce_operand_list(args)
    symbolic_args: list[OpValue] = []
    for i, a in enumerate(args_seq):
        if isinstance(a, Tensor):
            axis = _resolve_vmap_axis(in_axes, i)
            new_shape = _compute_vmap_shape(a, axis)
            proxy = ProxyTensor(id=str(uuid.uuid4()), shape=new_shape, dtype=a.dtype.value)
            symbolic_args.append(Tensor(proxy, TensorConfig(new_shape, a.dtype, a.device)))
        else:
            symbolic_args.append(a)
    return symbolic_args


@typing.runtime_checkable
class _TracedIdProtocol(typing.Protocol):
    """Protocol for traced proxy data exposing an ``id`` string."""

    @property
    def id(self) -> str:
        """Return traced node identifier.

        Returns:
            str: Traced node ID.
        """
        ...


def _extract_traced_id(val: OpValue) -> str:
    """Extract ``.id`` from traced tensor data or raise ``AttributeError``.

    Args:
        val (OpValue): Candidate traced tensor payload.

    Returns:
        str: Traced node identifier.

    Raises:
        AttributeError: If ``val`` does not expose an ``id`` attribute.
    """
    if isinstance(val, _TracedIdProtocol):
        return str(val.id)
    raise AttributeError(f"'{type(val).__name__}' object has no attribute 'id'")


def _trace_vmap(
    func: OpValue,
    in_axes: int | tuple[int, ...],
    out_axes: int | tuple[int, ...],
    args: Sequence[OpValue],
) -> OpValue:
    """Compute the trace vmap tensor operation.

    Args:
        func (OpValue): Input `func` argument (func).
        in_axes (int | tuple[int, ...]): Input `in_axes` argument (in axes).
        out_axes (int | tuple[int, ...]): Input `out_axes` argument (out axes).
        args (Sequence[OpValue]): Input `args` argument (args).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    symbolic_args = _create_vmap_symbolic_args(args, in_axes)
    body_graph = _trace_function(func, tuple(symbolic_args), "vmap_body")
    out_id = str(uuid.uuid4())
    node_inputs: list[str] = [_extract_traced_id(a.data) for a in args if isinstance(a, Tensor)]
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
    func: OpValue,
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
) -> OpValue:
    """Create a vectorized version of a function mapped over specified axes.

    Args:
        func (OpValue): Input `func` argument (func).
        in_axes (int | tuple[int, ...]): Input `in_axes` argument (in axes).
        out_axes (int | tuple[int, ...]): Input `out_axes` argument (out axes).

    Returns:
        Callable: Computed `Callable` output.
    """

    def wrapped(*args: OpValue) -> OpValue:
        """Wrap.

        Args:
            *args (OpValue): Positional args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

        Raises:
            RuntimeError: If execution or validation fails in `wrapped`.
        """
        if config.eager_mode:
            return _eager_vmap(func, in_axes, out_axes, args)
        if not global_tracing_state.is_tracing:
            msg = "Cannot emit Vmap outside of a tracing context."
            raise RuntimeError(msg)
        return _trace_vmap(func, in_axes, out_axes, args)

    return wrapped
