"""Unified Intermediate Representation (IR) Schema."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import (
    AttributeValue,
    LogicalGraph,
    LogicalNode,
    NodeDict,
    NoTangent,
    PartitionSpec,
    ZeroTangent,
)
from ml_switcheroo_ir import (
    LogicalGraph as IRGraph,
)
from ml_switcheroo_ir import (
    LogicalNode as IRNode,
)
from ml_switcheroo_ir.types import TensorSpec

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode

SymbolicDim = typing.Union[str, SymNode, SymInt]  # noqa: UP007
ShapeLike = typing.Union[tuple[typing.Union[int, str], ...], Sequence[typing.Union[int, str]], None]  # noqa: UP007
AxesLike = typing.Union[int, Sequence[int], None]  # noqa: UP007
ScalarLike = typing.Union[int, float, bool, complex]  # noqa: UP007
_AttrD1 = typing.Union[ScalarLike, str, bytes, DType, Device, IRGraph, None]  # noqa: UP007
_AttrD2 = typing.Union[_AttrD1, Sequence[_AttrD1], Mapping[str, _AttrD1]]  # noqa: UP007
AttrValue = typing.Union[_AttrD2, Sequence[_AttrD2], Mapping[str, _AttrD2]]  # noqa: UP007

# Re-export TensorSpec, ZeroTangent, and NoTangent from ml_switcheroo_ir
TensorSpec = TensorSpec
ZeroTangent = ZeroTangent
NoTangent = NoTangent

LogicalGraph.model_config["extra"] = "allow"
LogicalGraph.model_config["validate_assignment"] = False
LogicalGraph.model_rebuild(force=True)
LogicalNode.model_config["extra"] = "allow"
LogicalNode.model_config["validate_assignment"] = False
LogicalNode.model_rebuild(force=True)


def _is_logical_graph(val: TensorOrScalar) -> typing.TypeGuard[LogicalGraph]:
    """Type guard narrowing `val` to `LogicalGraph`.

    Args:
        val (TensorOrScalar): Candidate graph value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def ensure_graph(val: TensorOrScalar) -> LogicalGraph:
    """Narrow a runtime graph attribute or subgraph reference to `LogicalGraph`.

    Args:
        val (TensorOrScalar): Subgraph or graph attribute value.

    Returns:
        LogicalGraph: Narrowed LogicalGraph instance.
    """
    assert _is_logical_graph(val)
    return val


def _is_opt_logical_graph(val: TensorOrScalar) -> typing.TypeGuard[LogicalGraph | None]:
    """Type guard narrowing `val` to `LogicalGraph | None`.

    Args:
        val (TensorOrScalar): Candidate optional graph value.

    Returns:
        bool: Always True for optional graph values.
    """
    return val is None or isinstance(val, LogicalGraph) or not isinstance(val, type(Ellipsis))


def as_opt_graph(val: TensorOrScalar) -> LogicalGraph | None:
    """Narrow a runtime optional graph attribute to `LogicalGraph | None`.

    Args:
        val (TensorOrScalar): Optional subgraph or graph attribute value.

    Returns:
        LogicalGraph | None: Narrowed LogicalGraph or None.
    """
    assert _is_opt_logical_graph(val)
    return val


def _is_attr_dict(val: TensorOrScalar) -> typing.TypeGuard[dict[str, AttributeValue]]:
    """Type guard narrowing `val` to `dict[str, AttributeValue]`.

    Args:
        val (TensorOrScalar): Candidate attribute dictionary.

    Returns:
        bool: Always True for attribute dictionaries.
    """
    return isinstance(val, dict) or not isinstance(val, type(Ellipsis))


def ensure_attrs(val: TensorOrScalar) -> dict[str, AttributeValue]:
    """Narrow a dictionary of node attributes to `dict[str, AttributeValue]`.

    Args:
        val (TensorOrScalar): Dictionary of attributes.

    Returns:
        dict[str, AttributeValue]: Narrowed attribute dictionary.
    """
    assert _is_attr_dict(val)
    return val


def _is_attr_val(val: TensorOrScalar) -> typing.TypeGuard[AttributeValue]:
    """Type guard narrowing `val` to `AttributeValue`.

    Args:
        val (TensorOrScalar): Candidate attribute value.

    Returns:
        bool: Always True for attribute values.
    """
    return not isinstance(val, type(Ellipsis))


def ensure_attr_val(val: TensorOrScalar) -> AttributeValue:
    """Narrow a runtime attribute value to `AttributeValue`.

    Args:
        val (TensorOrScalar): Input attribute value.

    Returns:
        AttributeValue: Narrowed attribute value.
    """
    assert _is_attr_val(val)
    return val


def _is_node_list(val: TensorOrScalar) -> typing.TypeGuard[list[LogicalNode]]:
    """Type guard narrowing `val` to `list[LogicalNode]`.

    Args:
        val (TensorOrScalar): Candidate node list.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def as_node_list(val: TensorOrScalar) -> list[LogicalNode]:
    """Narrow a sequence or iterable of IR nodes to `list[LogicalNode]`.

    Args:
        val (TensorOrScalar): Iterable of IR nodes.

    Returns:
        list[LogicalNode]: List of LogicalNode objects.
    """
    assert _is_node_list(val)
    return val


def _is_opt_pspec(val: TensorOrScalar) -> typing.TypeGuard[PartitionSpec | None]:
    """Type guard narrowing `val` to `PartitionSpec | None`.

    Args:
        val (TensorOrScalar): Candidate optional partition spec.

    Returns:
        bool: Always True for optional partition specs.
    """
    return val is None or isinstance(val, PartitionSpec) or not isinstance(val, type(Ellipsis))


def as_opt_pspec(val: TensorOrScalar) -> PartitionSpec | None:
    """Narrow an optional partition spec attribute to `PartitionSpec | None`.

    Args:
        val (TensorOrScalar): Input partition spec or None.

    Returns:
        PartitionSpec | None: Narrowed PartitionSpec or None.
    """
    assert _is_opt_pspec(val)
    return val


__all__ = [
    "AttrValue",
    "AxesLike",
    "DType",
    "IRGraph",
    "IRNode",
    "LogicalGraph",
    "LogicalNode",
    "NoTangent",
    "NodeDict",
    "ScalarLike",
    "ShapeLike",
    "SymbolicDim",
    "TensorSpec",
    "ZeroTangent",
    "as_opt_graph",
    "clone_logical_node",
    "ensure_attr_val",
    "ensure_attrs",
    "ensure_graph",
]


LogicalNodeOverrideValue = typing.Union[
    "str",
    int,
    dict[str, AttributeValue],
    list[str],
    Sequence[typing.Union[int, "str"]],
    PartitionSpec,
    DType,
    list[TensorSpec],
    dict[str, LogicalGraph],
    None,
]


def clone_logical_node(node: LogicalNode, **overrides: LogicalNodeOverrideValue) -> LogicalNode:
    """Clones a LogicalNode, allowing overrides via keyword arguments.

    Args:
        node (LogicalNode): The source LogicalNode to clone.
        **overrides (LogicalNodeOverrideValue): Keyword overrides for LogicalNode fields.

    Returns:
        LogicalNode: The cloned LogicalNode.
    """
    return node.model_copy(update=dict(overrides), deep=True)
