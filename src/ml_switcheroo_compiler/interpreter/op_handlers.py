"""Subgraph structural protocols and specialized IR operation handlers for the interpreter."""

from __future__ import annotations

import builtins
import types
import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import BackendOperand, ShapeProtocol, coerce_operand_list
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.interpreter.environment import Environment, InterpreterValue, as_eval_value
from ml_switcheroo_compiler.interpreter.slice_parser import (
    SliceKey,
    _eval_getitem,
    _eval_setitem,
    _eval_to_int,
    _parse_slice_string,
)


@typing.runtime_checkable
class _ScalarItemProtocol(typing.Protocol):
    """Protocol for scalar tensor values exposing `.item()`."""

    def item(self) -> int | float | bool | complex | str | bytes:
        """Return scalar value.

        Returns:
            int | float | bool | complex | str | bytes: Scalar value.
        """
        ...


@typing.runtime_checkable
class _ExecuteOpBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `execute_op`."""

    def execute_op(self, op_name: str, *args: InterpreterValue, **kwargs: AttributeValue) -> OpValue:
        """Execute an operation on the backend.

        Args:
            op_name (str): Operation name.
            *args (InterpreterValue): Positional arguments.
            **kwargs (AttributeValue): Keyword arguments.

        Returns:
            OpValue: Operation output.
        """
        ...


InterpreterBackend = typing.Union[type[BaseGenerator], types.ModuleType, BackendOperand]  # noqa: UP007


@typing.runtime_checkable
class _CustomNodeFieldsProtocol(typing.Protocol):
    """Protocol for custom node objects exposing `LogicalNode` fields."""

    id: str
    op_type: str
    inputs: list[str]
    attributes: dict[str, AttributeValue]


@typing.runtime_checkable
class _SubgraphNodesProtocol(typing.Protocol):
    """Protocol for subgraphs exposing `nodes`."""

    nodes: Mapping[str, LogicalNode | _CustomNodeFieldsProtocol] | Sequence[LogicalNode | _CustomNodeFieldsProtocol]


@typing.runtime_checkable
class _SubgraphInputsProtocol(typing.Protocol):
    """Protocol for subgraphs exposing `inputs`."""

    inputs: list[str]


@typing.runtime_checkable
class _SubgraphOutputsProtocol(typing.Protocol):
    """Protocol for subgraphs exposing `outputs`."""

    outputs: list[str] | str


@typing.runtime_checkable
class _NamedSubgraphProtocol(typing.Protocol):
    """Protocol for subgraphs exposing `id`."""

    id: str


SubgraphCandidate = typing.Union[  # noqa: UP007
    OpValue,
    InterpreterValue,
    _NamedSubgraphProtocol,
    _SubgraphInputsProtocol,
    _SubgraphOutputsProtocol,
    _SubgraphNodesProtocol,
]


def _has_subgraph_nodes(val: SubgraphCandidate) -> typing.TypeGuard[_SubgraphNodesProtocol]:
    """Check whether `val` exposes `nodes`.

    Args:
        val (SubgraphCandidate): Candidate subgraph object.

    Returns:
        bool: True if `val` has `nodes`.
    """
    return isinstance(val, _SubgraphNodesProtocol)


def _has_subgraph_inputs(val: SubgraphCandidate) -> typing.TypeGuard[_SubgraphInputsProtocol]:
    """Check whether `val` exposes `inputs`.

    Args:
        val (SubgraphCandidate): Candidate subgraph object.

    Returns:
        bool: True if `val` has `inputs`.
    """
    return isinstance(val, _SubgraphInputsProtocol)


def _has_subgraph_outputs(val: SubgraphCandidate) -> typing.TypeGuard[_SubgraphOutputsProtocol]:
    """Check whether `val` exposes `outputs`.

    Args:
        val (SubgraphCandidate): Candidate subgraph object.

    Returns:
        bool: True if `val` has `outputs`.
    """
    return isinstance(val, _SubgraphOutputsProtocol)


def _has_named_subgraph_id(val: SubgraphCandidate) -> typing.TypeGuard[_NamedSubgraphProtocol]:
    """Check whether `val` exposes `id`.

    Args:
        val (SubgraphCandidate): Candidate subgraph object.

    Returns:
        bool: True if `val` has `id`.
    """
    return isinstance(val, _NamedSubgraphProtocol)


def _has_scalar_item(val: OpValue | InterpreterValue | _ScalarItemProtocol) -> typing.TypeGuard[_ScalarItemProtocol]:
    """Check whether `val` exposes `item`.

    Args:
        val (OpValue | InterpreterValue | _ScalarItemProtocol): Candidate scalar array.

    Returns:
        bool: True if `val` has `item`.
    """
    return isinstance(val, _ScalarItemProtocol)


def _has_execute_op(val: InterpreterBackend) -> typing.TypeGuard[_ExecuteOpBackendProtocol]:
    """Check whether `val` exposes `execute_op`.

    Args:
        val (InterpreterBackend): Active backend.

    Returns:
        bool: True if `val` has `execute_op`.
    """
    return type(val) is not types.ModuleType and isinstance(val, _ExecuteOpBackendProtocol)


def _has_eval_shape(val: OpValue | InterpreterValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes `shape`.

    Args:
        val (OpValue | InterpreterValue): Candidate tensor or array.

    Returns:
        bool: True if `val` has `shape`.
    """
    return isinstance(val, ShapeProtocol)


def _coerce_logical_node(n: LogicalNode | _CustomNodeFieldsProtocol) -> LogicalNode:
    """Return `n` if it is a `LogicalNode`, or construct one from its fields.

    Args:
        n (LogicalNode | _CustomNodeFieldsProtocol): Node instance.

    Returns:
        LogicalNode: Coerced `LogicalNode`.
    """
    return (
        n
        if isinstance(n, LogicalNode)
        else LogicalNode(id=n.id, op_type=n.op_type, inputs=list(n.inputs), attributes=dict(n.attributes or {}))
    )


def _extract_subgraph_nodes(subgraph: SubgraphCandidate) -> list[LogicalNode]:
    """Extract ordered `LogicalNode` list from a subgraph's `.nodes` container.

    Args:
        subgraph (SubgraphCandidate): Subgraph instance exposing `.nodes`.

    Returns:
        list[LogicalNode]: Extracted nodes list.
    """
    assert _has_subgraph_nodes(subgraph)
    raw_nodes = list(subgraph.nodes.values()) if isinstance(subgraph.nodes, Mapping) else list(subgraph.nodes)
    return [_coerce_logical_node(n) for n in raw_nodes]


def _eval_truthy(val: OpValue | InterpreterValue | _ScalarItemProtocol) -> bool:
    """Evaluate the boolean truthiness of a scalar or 0-D tensor value.

    Args:
        val (OpValue | InterpreterValue | _ScalarItemProtocol): Condition value.

    Returns:
        bool: Boolean truth value.
    """
    if isinstance(val, (bool, int, float)):
        return bool(val)
    return bool(val.item()) if _has_scalar_item(val) else bool(val)


def _handle_slice(
    node: LogicalNode,
    env: Environment,
    backend: InterpreterBackend,
    in_vals: Sequence[InterpreterValue],
    kwargs: Mapping[str, AttributeValue],
) -> None:
    """Execute a slicing operation on a tensor and store the result.

    Args:
        node (LogicalNode): The IR node representing the slice operation.
        env (Environment): The current local variable environment for storing results.
        backend (InterpreterBackend): The active compute backend used for array manipulation.
        in_vals (Sequence[InterpreterValue]): Input tensor values to be sliced.
        kwargs (Mapping[str, AttributeValue]): Slice keyword arguments (`slices`, `dim`, `start`, `end`, `step`).
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    arr: OpValue = adapt_backend_module(backend).require_op("asarray")(in_vals[0])
    if "slices" in kwargs:
        parsed_key = _parse_slice_string(str(kwargs["slices"]))
        env.set(node.id, _eval_getitem(arr, parsed_key))
        return

    raw_dim = kwargs.get("dim", 0)
    dim = _eval_to_int(raw_dim, 0) if raw_dim is not None else 0
    raw_start = kwargs.get("start", None)
    raw_end = kwargs.get("end", None)
    raw_step = kwargs.get("step", 1)
    start: int | float | str | None = raw_start if isinstance(raw_start, (int, float, str)) else None
    end: int | float | str | None = raw_end if isinstance(raw_end, (int, float, str)) else None
    step: int | float | str | None = raw_step if isinstance(raw_step, (int, float, str)) else None

    first_in = in_vals[0]
    in_rank = len(first_in.shape) if _has_eval_shape(first_in) else 1
    sl: list[SliceKey] = [builtins.slice(None)] * in_rank
    sl[dim] = builtins.slice(start, end, step)
    env.set(node.id, _eval_getitem(arr, tuple(sl)))


def _handle_getitem(
    node: LogicalNode,
    env: Environment,
    backend: InterpreterBackend,
    in_vals: Sequence[InterpreterValue],
    kwargs: Mapping[str, AttributeValue],
) -> None:
    """Execute a `GetItem` indexing operation and store the result.

    Args:
        node (LogicalNode): The IR node representing the getitem operation.
        env (Environment): The local variable environment.
        backend (InterpreterBackend): The active compute backend for array manipulation.
        in_vals (Sequence[InterpreterValue]): Input tensor sequence.
        kwargs (Mapping[str, AttributeValue]): Keyword arguments containing the `key` expression.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    key = kwargs.get("key")
    parsed_key = _parse_slice_string(str(key))
    arr: OpValue = adapt_backend_module(backend).require_op("asarray")(in_vals[0])
    env.set(node.id, _eval_getitem(arr, parsed_key))


def _handle_index_put(
    node: LogicalNode,
    env: Environment,
    backend: InterpreterBackend,
    in_vals: Sequence[InterpreterValue],
    kwargs: Mapping[str, AttributeValue],
) -> None:
    """Execute an `IndexPut` / `SetItem` operation placing update elements into a cloned tensor.

    Args:
        node (LogicalNode): The IR node representing the index_put operation.
        env (Environment): The local variable environment.
        backend (InterpreterBackend): The active compute backend for array manipulation.
        in_vals (Sequence[InterpreterValue]): Base tensor and update value sequence.
        kwargs (Mapping[str, AttributeValue]): Keyword arguments containing target `key` or `index`.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    key = kwargs.get("key", kwargs.get("index", 0))
    parsed_key = _parse_slice_string(str(key))
    arr = as_eval_value(adapt_backend_module(backend).require_op("execute_op")("Clone", in_vals[0]))
    val = in_vals[1]
    _eval_setitem(arr, parsed_key, val)
    env.set(node.id, arr)


def _handle_checkpoint(
    node: LogicalNode,
    env: Environment,
    in_vals: Sequence[InterpreterValue],
) -> None:
    """Execute a gradient checkpointing subgraph dynamically.

    Args:
        node (LogicalNode): The IR node representing the checkpoint wrapper.
        env (Environment): The local variable environment.
        in_vals (Sequence[InterpreterValue]): Input tensors to feed into the subgraph.

    Raises:
        AttributeError: If the checkpoint node is missing a valid subgraph.
    """
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    raw_sg: SubgraphCandidate = (
        node.subgraphs.get("body") or node.subgraphs.get("subgraph") or node.attributes.get("subgraph")
    )
    raw_sg_out: SubgraphCandidate = raw_sg
    if not _has_subgraph_inputs(raw_sg) or not _has_subgraph_outputs(raw_sg_out):
        msg = "Checkpoint node is missing a valid subgraph"
        raise AttributeError(msg)
    sg_inputs = list(raw_sg.inputs)
    raw_outputs = raw_sg_out.outputs
    sg_outputs = list(raw_outputs) if isinstance(raw_outputs, list) else [str(raw_outputs)]
    sub_inputs: dict[str, InterpreterValue] = dict(zip(sg_inputs, in_vals))
    sub_graph = LogicalGraph(name="checkpoint_subgraph")
    for n in _extract_subgraph_nodes(raw_sg):
        sub_graph.nodes[n.id] = n
    sub_graph.inputs = sg_inputs
    sub_graph.outputs = sg_outputs

    sub_outputs = evaluate_graph(sub_graph, sub_inputs)

    if len(sub_graph.outputs) == 1:
        result: InterpreterValue = sub_outputs[sub_graph.outputs[0]]
    else:
        result = tuple(sub_outputs[out_id] for out_id in sub_graph.outputs)
    env.set(node.id, result)


def _handle_meshgrid(
    node: LogicalNode,
    env: Environment,
    backend: InterpreterBackend,
    in_vals: Sequence[InterpreterValue],
    kwargs: dict[str, AttributeValue],
) -> None:
    """Execute a `Meshgrid` operation and extract the requested output index.

    Args:
        node (LogicalNode): The IR node representing the meshgrid operation.
        env (Environment): The local variable environment.
        backend (InterpreterBackend): The active compute backend for execution.
        in_vals (Sequence[InterpreterValue]): Input 1D coordinate arrays.
        kwargs (dict[str, AttributeValue]): Keyword arguments including `output_index`.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    raw_idx = kwargs.pop("output_index", 0)
    idx = int(raw_idx) if isinstance(raw_idx, (int, float, str)) else 0
    result = adapt_backend_module(backend).require_op("execute_op")("Meshgrid", *in_vals, **kwargs)
    env.set(node.id, as_eval_value(coerce_operand_list(result)[idx]))


def _get_op_alias(op_type: str) -> str:
    """Resolve an operation alias to its standardized backend operation name.

    Args:
        op_type (str): The raw operation name.

    Returns:
        str: The standardized operation name.
    """
    op_alias = {
        "Sub": "Subtract",
        "Mul": "Multiply",
        "Div": "TrueDivide",
        "Neg": "Negative",
        "Pow": "Power",
        "MatMul": "Matmul",
        "Expand": "BroadcastTo",
        "Permute": "Transpose",
    }
    return op_alias.get(op_type, op_type)


def _prepare_node_kwargs(node: LogicalNode, target_op: str) -> dict[str, AttributeValue]:
    """Prepare the keyword arguments required for executing a logical node.

    Args:
        node (LogicalNode): The target node.
        target_op (str): The resolved operation name.

    Returns:
        dict[str, AttributeValue]: Mapping of argument names to serializable IR attribute values.
    """
    kwargs: dict[str, AttributeValue] = {
        k: v for k, v in node.attributes.items() if k not in ("var_name", "is_parameter", "param_name")
    }
    if "strides" in kwargs and target_op not in {
        "Conv",
        "Conv1D",
        "Conv2D",
        "Conv3D",
        "ConvTranspose",
        "ConvTranspose2D",
        "MaxPool",
        "MaxPool1D",
        "MaxPool2D",
        "MaxPool3D",
        "AvgPool",
        "AvgPool1D",
        "AvgPool2D",
        "AvgPool3D",
        "Pool1D",
        "Pool2D",
        "Pool3D",
        "StridedSlice",
    }:
        kwargs.pop("strides", None)
    if node.shape_metadata:
        meta_seq = node.shape_metadata or ()
        meta_tuple: tuple[int | str, ...] = tuple(int(d) if isinstance(d, int) else str(d) for d in meta_seq)
        if target_op in ("Expand", "BroadcastTo", "ConstantOfShape", "Zeros", "Ones", "Full") and "shape" not in kwargs:
            kwargs["shape"] = meta_tuple
        if target_op == "Reshape" and "newshape" not in kwargs:
            kwargs["newshape"] = meta_tuple
    return kwargs
