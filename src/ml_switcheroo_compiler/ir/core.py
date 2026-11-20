"""Compiler-local narrowing, attribute normalization, and node cloning helpers for Unified IR."""

from __future__ import annotations

import copy
import typing
from collections.abc import Mapping, Sequence

import pydantic
from ml_switcheroo_ir import (
    ArrayBufferProtocol,
    AttributeValue,
    CustomRuleHandleProtocol,
    DeviceLikeProtocol,
    DimensionType,
    DTypeLikeProtocol,
    IRGraph,
    LogicalGraph,
    LogicalNode,
    LoopOptionsProtocol,
    PartitionSpec,
    SymbolicIntWrapperProtocol,
    SymbolicNodeProtocol,
    TensorShape,
    TensorSpec,
)
from ml_switcheroo_ir import DType as IRDType
from ml_switcheroo_ir.graph import CloneOverrideValue

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core._shape_protocols import ShapeDim
    from ml_switcheroo_compiler.core.shape import OpValue

SymbolicDim = typing.Union[str, SymNode, SymInt]  # noqa: UP007
ShapeLike = typing.Union[tuple[typing.Union[int, str], ...], Sequence[typing.Union[int, str]], None]  # noqa: UP007
AxesLike = typing.Union[int, Sequence[int], None]  # noqa: UP007
ScalarLike = typing.Union[int, float, bool, complex]  # noqa: UP007
_AttrD1 = typing.Union[ScalarLike, str, bytes, DType, IRDType, Device, IRGraph, None]  # noqa: UP007
_AttrD2 = typing.Union[_AttrD1, Sequence[_AttrD1], Mapping[str, _AttrD1]]  # noqa: UP007
IRAttrValue = typing.Union[_AttrD2, Sequence[_AttrD2], Mapping[str, _AttrD2]]  # noqa: UP007


def ensure_graph(val: OpValue) -> LogicalGraph:
    """Validate and return a ``LogicalGraph`` instance or construct an empty default when ``None``.

    Args:
        val (OpValue): Subgraph, graph container, or ``None``.

    Returns:
        LogicalGraph: Validated ``LogicalGraph`` instance.

    Raises:
        TypeError: If ``val`` is neither ``None`` nor a ``LogicalGraph`` object.
    """
    if val is None:
        return LogicalGraph()
    if isinstance(val, LogicalGraph):
        return val
    raise TypeError(f"Expected LogicalGraph, got {type(val).__name__}.")


def as_opt_graph(val: OpValue) -> LogicalGraph | None:
    """Narrow an optional graph attribute to ``LogicalGraph | None``.

    Args:
        val (OpValue): Optional subgraph or graph attribute value.

    Returns:
        LogicalGraph | None: ``LogicalGraph`` when present, otherwise ``None``.
    """
    if isinstance(val, LogicalGraph):
        return val
    return None


def ensure_attrs(val: Mapping[str, AttributeValue] | Mapping[str, OpValue] | OpValue) -> dict[str, AttributeValue]:
    """Normalize a node attribute mapping into a mutable ``dict[str, AttributeValue]``.

    Args:
        val (Mapping[str, AttributeValue] | Mapping[str, OpValue] | OpValue): Mapping or dictionary of attributes.

    Returns:
        dict[str, AttributeValue]: Mutable attribute dictionary.
    """
    if isinstance(val, dict):
        return {str(k): ensure_attr_val(v) for k, v in val.items()}
    if isinstance(val, Mapping):
        return {str(k): ensure_attr_val(v) for k, v in val.items()}
    return {}


def ensure_attr_val(val: OpValue) -> AttributeValue:
    """Validate and normalize ``val`` into a serializable IR ``AttributeValue``.

    Args:
        val (OpValue): Input attribute value.

    Returns:
        AttributeValue: Validated attribute value accepted by ``ml_switcheroo_ir.LogicalNode``.
    """
    if val is None or isinstance(
        val,
        (
            int,
            float,
            bool,
            str,
            bytes,
            IRDType,
            DTypeLikeProtocol,
            DeviceLikeProtocol,
            TensorShape,
            TensorSpec,
            PartitionSpec,
            SymbolicNodeProtocol,
            SymbolicIntWrapperProtocol,
            CustomRuleHandleProtocol,
            LoopOptionsProtocol,
            ArrayBufferProtocol,
            pydantic.BaseModel,
        ),
    ):
        return val
    if isinstance(val, tuple):
        return tuple(ensure_attr_val(x) for x in val)
    if isinstance(val, Sequence):
        return [ensure_attr_val(x) for x in val]
    if isinstance(val, Mapping):
        return {str(k): ensure_attr_val(v) for k, v in val.items()}
    from ml_switcheroo_compiler.core.shape import is_variadic_callable
    from ml_switcheroo_compiler.ir.callback_registry import register_custom_rule_handle

    if not isinstance(val, type) and is_variadic_callable(val):
        handle = register_custom_rule_handle(val)
        assert handle is not None
        return handle
    return str(val)


def as_node_list(val: OpValue) -> list[LogicalNode]:
    """Normalize a mapping or sequence of IR nodes into ``list[LogicalNode]``.

    Args:
        val (OpValue): Mapping or sequence of ``LogicalNode`` objects.

    Returns:
        list[LogicalNode]: Extracted list of ``LogicalNode`` instances.
    """
    if isinstance(val, Mapping):
        return [v for v in val.values() if isinstance(v, LogicalNode)]
    if isinstance(val, list):
        return [v for v in val if isinstance(v, LogicalNode)]
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        return [v for v in val if isinstance(v, LogicalNode)]
    return []


def as_opt_pspec(val: OpValue) -> PartitionSpec | None:
    """Narrow an optional sharding attribute to ``PartitionSpec | None``.

    Args:
        val (OpValue): Input ``PartitionSpec`` or ``None``.

    Returns:
        PartitionSpec | None: ``PartitionSpec`` when present, otherwise ``None``.
    """
    if isinstance(val, PartitionSpec):
        return val
    return None


def normalize_shape_metadata(
    shape: Sequence[ShapeDim | DimensionType] | None,
) -> tuple[DimensionType, ...] | None:
    """Normalize compiler ``ShapeDim`` sequence (where ``None`` denotes dynamic dim) into ``DimensionType`` tuple.

    Args:
        shape (Sequence[ShapeDim | DimensionType] | None): Compiler shape sequence or ``None``.

    Returns:
        tuple[DimensionType, ...] | None: Normalized IR shape metadata tuple or ``None``.
    """
    if shape is None:
        return None
    return tuple(-1 if d is None else d for d in shape)


def _clone_graph(graph: LogicalGraph) -> LogicalGraph:
    """Recursively clone a ``LogicalGraph`` and all nested ``LogicalNode`` / subgraph attributes.

    Args:
        graph (LogicalGraph): Source ``LogicalGraph`` instance.

    Returns:
        LogicalGraph: Isolated deep clone of ``graph``.
    """
    cloned_nodes = {nid: clone_logical_node(n) for nid, n in graph.nodes.items()}
    cloned_sorted = (
        [cloned_nodes[n.id] if n.id in cloned_nodes else clone_logical_node(n) for n in graph.sorted_nodes]
        if graph.sorted_nodes is not None
        else None
    )
    return LogicalGraph(
        name=graph.name,
        nodes=cloned_nodes,
        inputs=list(graph.inputs),
        input_specs={k: copy.deepcopy(v) for k, v in graph.input_specs.items()},
        outputs=list(graph.outputs),
        mesh=copy.deepcopy(graph.mesh) if graph.mesh is not None else None,
        initializers=copy.deepcopy(graph.initializers),
        pipeline_topology=copy.deepcopy(graph.pipeline_topology) if graph.pipeline_topology is not None else None,
        attributes={k: _clone_attr_value(v) for k, v in graph.attributes.items()},
        annotations={k: copy.deepcopy(v) for k, v in graph.annotations.items()},
        metadata={k: _clone_attr_value(v) for k, v in graph.metadata.items()},
        sorted_nodes=cloned_sorted,
        loop_options=copy.deepcopy(graph.loop_options) if graph.loop_options is not None else None,
    )


def _clone_attr_value(val: AttributeValue) -> AttributeValue:
    """Recursively clone an ``AttributeValue`` while using ``_clone_graph`` on ``LogicalGraph`` / ``LogicalNode``.

    Args:
        val (AttributeValue): Source attribute value.

    Returns:
        AttributeValue: Deeply cloned attribute value safe from ``NodeDict`` reconstruction cycles.
    """
    if isinstance(val, LogicalGraph):
        return _clone_graph(val)
    if isinstance(val, LogicalNode):
        return clone_logical_node(val)
    if isinstance(val, dict):
        return {k: _clone_attr_value(v) for k, v in val.items()}
    if isinstance(val, tuple):
        return tuple(_clone_attr_value(x) for x in val)
    if isinstance(val, list):
        return [_clone_attr_value(x) for x in val]
    return copy.deepcopy(val)


def clone_logical_node(
    node: LogicalNode,
    id: str | None = None,
    op_type: str | None = None,
    **overrides: CloneOverrideValue,
) -> LogicalNode:
    """Clone a ``LogicalNode`` with isolated containers and optional field overrides.

    Args:
        node (LogicalNode): Source ``LogicalNode`` to clone.
        id (str | None): Optional replacement node identifier.
        op_type (str | None): Optional replacement operation type.
        **overrides (CloneOverrideValue): Keyword overrides forwarded to ``LogicalNode.clone``.

    Returns:
        LogicalNode: Cloned ``LogicalNode`` instance.
    """
    raw_attrs = overrides.pop("attributes", None)
    raw_subs = overrides.pop("subgraphs", None)
    src_attrs = raw_attrs if isinstance(raw_attrs, Mapping) else node.attributes
    cloned_attrs: dict[str, AttributeValue] = {str(k): _clone_attr_value(v) for k, v in src_attrs.items()}

    src_subs = raw_subs if isinstance(raw_subs, Mapping) else node.subgraphs
    cloned_subs: dict[str, LogicalGraph | AttributeValue] = {
        str(k): (_clone_graph(v) if isinstance(v, LogicalGraph) else _clone_attr_value(v)) for k, v in src_subs.items()
    }

    cloned = node.clone(id=id, op_type=op_type, attributes={}, subgraphs={}, **overrides)
    if cloned_attrs or cloned.attributes:
        merged_attrs = dict(cloned_attrs)
        merged_attrs.update(cloned.attributes)
        cloned.attributes = merged_attrs
    if cloned_subs:
        cloned.subgraphs = cloned_subs
    return cloned


__all__ = [
    "AxesLike",
    "IRAttrValue",
    "ScalarLike",
    "ShapeLike",
    "SymbolicDim",
    "as_node_list",
    "as_opt_graph",
    "as_opt_pspec",
    "clone_logical_node",
    "ensure_attr_val",
    "ensure_attrs",
    "ensure_graph",
    "normalize_shape_metadata",
]
