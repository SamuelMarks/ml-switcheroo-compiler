"""Graph tracing and backend execution helpers for JVP and VJP transformations."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Iterable, Mapping

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.shape import OpValue, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.interpreter.environment import InterpreterValue, as_eval_value
from ml_switcheroo_compiler.ir.core import as_node_list
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function

from .custom_jvp_ops import (
    _has_id_data_attr,
)
from .utils import (
    PrimalTree,
    _unwrap_data,
)


@typing.runtime_checkable
class _ToListProtocol(typing.Protocol):
    """Protocol for arrays supporting `.tolist()`."""

    def tolist(self) -> OpValue:
        """Convert array to nested Python list.

        Returns:
            OpValue: Nested Python list.
        """
        ...


@typing.runtime_checkable
class _ExecuteGraphBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `execute_graph`."""

    def execute_graph(self, graph: LogicalGraph, inputs_dict: dict[str, InterpreterValue]) -> Mapping[str, OpValue]:
        """Execute graph with inputs.

        Args:
            graph (LogicalGraph): Target graph.
            inputs_dict (dict[str, InterpreterValue]): Concrete inputs mapping.

        Returns:
            Mapping[str, OpValue]: Outputs mapping.
        """
        ...


@typing.runtime_checkable
class _CompileGraphBackendProtocol(typing.Protocol):
    """Protocol for backends implementing `compile_graph`."""

    def compile_graph(self, graph: LogicalGraph) -> TypedVariadicCallable[Mapping[str, OpValue]]:
        """Compile graph into a callable.

        Args:
            graph (LogicalGraph): Target graph.

        Returns:
            TypedVariadicCallable[Mapping[str, OpValue]]: Compiled callable.
        """
        ...


@typing.runtime_checkable
class _IdBlockProtocol(typing.Protocol):
    """Protocol for traced blocks exposing `.id`."""

    id: str


def _has_tolist_method(val: OpValue) -> typing.TypeGuard[_ToListProtocol]:
    """Check whether `val` exposes `tolist`.

    Args:
        val (OpValue): Candidate array.

    Returns:
        bool: True if `val` has `tolist`.
    """
    return isinstance(val, _ToListProtocol)


def _is_execute_graph_backend(val: OpValue) -> typing.TypeGuard[_ExecuteGraphBackendProtocol]:
    """Check whether `val` exposes `execute_graph`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `execute_graph`.
    """
    return isinstance(val, _ExecuteGraphBackendProtocol)


def _is_compile_graph_backend(val: OpValue) -> typing.TypeGuard[_CompileGraphBackendProtocol]:
    """Check whether `val` exposes `compile_graph`.

    Args:
        val (OpValue): Active backend.

    Returns:
        bool: True if `val` has `compile_graph`.
    """
    return isinstance(val, _CompileGraphBackendProtocol)


def _has_id_block_attr(val: OpValue) -> typing.TypeGuard[_IdBlockProtocol]:
    """Check whether `val` exposes `id`.

    Args:
        val (OpValue): Traced block or data.

    Returns:
        bool: True if `val` has `id`.
    """
    return isinstance(val, _IdBlockProtocol)


def _to_list_or_self(val: OpValue | PrimalTree) -> OpValue:
    """Return `val.tolist()` if `val` supports `.tolist()`, otherwise `val`.

    Args:
        val (OpValue | PrimalTree): Candidate array or scalar.

    Returns:
        OpValue: Python list or original value.
    """
    raw: OpValue = val
    return raw.tolist() if _has_tolist_method(raw) else raw


def _extract_node_id(val: OpValue | PrimalTree) -> str:
    """Return `val.data.id` if `val` has a traced `.data.id`, else `str(val.data)`.

    Args:
        val (OpValue | PrimalTree): Traced tensor or array.

    Returns:
        str: Node identifier string.
    """
    p_data = _unwrap_data(val)
    raw_data: OpValue = p_data
    return str(raw_data.id) if _has_id_data_attr(raw_data) else str(p_data)


def _unpack_pair(val: OpValue | PrimalTree) -> tuple[OpValue, OpValue]:
    """Unpack a 2-element iterable or tuple into `(first, second)`.

    Args:
        val (OpValue | PrimalTree): 2-element sequence or iterable.

    Returns:
        tuple[OpValue, OpValue]: Unpacked pair of elements.
    """
    assert isinstance(val, Iterable)
    seq = list(val)
    return seq[0], seq[1]


def _build_traced_logical_graph(
    fun_primal: TypedVariadicCallable[PrimalTree],
    tensor_primals: list[Tensor],
    prefix: str,
) -> LogicalGraph:
    """Trace a primal function and construct a `LogicalGraph` from the traced block.

    Args:
        fun_primal (TypedVariadicCallable[PrimalTree]): Primal function to trace.
        tensor_primals (list[Tensor]): Tensor primal inputs.
        prefix (str): Name prefix for the traced graph.

    Returns:
        LogicalGraph: Populated forward logical graph.
    """
    block = _trace_function(fun_primal, tuple(tensor_primals), f"{prefix}_{uuid.uuid4().hex[:6]}")
    raw_b: OpValue = block
    block_name = str(raw_b.id) if _has_id_block_attr(raw_b) and raw_b.id else (block.name or prefix)
    forward_graph = LogicalGraph(name=block_name)
    raw_nodes: OpValue = block.nodes
    nodes_iter: Iterable[LogicalNode] = (
        block.nodes.values() if isinstance(raw_nodes, Mapping) else as_node_list(block.nodes)
    )
    for node in nodes_iter:
        forward_graph.nodes[node.id] = node
    forward_graph.inputs = block.inputs
    forward_graph.outputs = block.outputs
    return forward_graph


def _execute_autodiff_graph(
    graph: LogicalGraph, inputs_dict: dict[str, InterpreterValue]
) -> dict[str, InterpreterValue]:
    """Execute an autodiff graph on the active backend or fallback interpreter.

    Args:
        graph (LogicalGraph): The JVP or VJP logical graph to execute.
        inputs_dict (dict[str, InterpreterValue]): Mapping of input node IDs to concrete arrays.

    Returns:
        dict[str, InterpreterValue]: Mapping of output node IDs to evaluated results.
    """
    import ml_switcheroo_compiler.grad.jvp_vjp as _jv_mod
    from ml_switcheroo_compiler.interpreter import evaluate_graph

    backend: OpValue = _jv_mod.get_active_backend()
    if _is_execute_graph_backend(backend):
        return {k: as_eval_value(v) for k, v in backend.execute_graph(graph, inputs_dict).items()}
    if _is_compile_graph_backend(backend):
        compiled_fn = backend.compile_graph(graph)
        return {k: as_eval_value(v) for k, v in compiled_fn(inputs_dict).items()}
    return evaluate_graph(graph, inputs_dict)
