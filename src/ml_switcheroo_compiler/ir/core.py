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
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode

SymbolicDim = typing.Union[str, "SymNode", "SymInt"]
ShapeLike = typing.Union[tuple[typing.Union[int, "str"], ...], Sequence[typing.Union[int, "str"]], "None"]
AxesLike = typing.Union[int, Sequence[int], "None"]
ScalarLike = typing.Union[int, float, bool, "complex"]
AttrValue = typing.Union[
    ScalarLike,
    "str",
    bytes,
    Sequence["AttrValue"],
    Mapping[str, "AttrValue"],
    DType,
    Device,
    IRGraph,
    None,
]

# Re-export TensorSpec, ZeroTangent, and NoTangent from ml_switcheroo_ir
TensorSpec = TensorSpec
ZeroTangent = ZeroTangent
NoTangent = NoTangent

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
    "clone_logical_node",
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


def clone_logical_node(
    node: LogicalNode,
    **overrides: LogicalNodeOverrideValue,
) -> LogicalNode:
    """Clones a LogicalNode, allowing overrides via keyword arguments.

    Args:
        node (LogicalNode): The source LogicalNode to clone.
        **overrides (LogicalNodeOverrideValue): Optional field overrides (id, op_type, domain,
            version, attributes, inputs, outputs, shape_metadata, source_ast_ref, sharding,
            dtype, output_specs, subgraphs, device, stream).

    Returns:
        LogicalNode: The cloned LogicalNode.
    """
    clone_fn = typing.cast(TypedVariadicCallable[LogicalNode], getattr(node, "clone"))
    return clone_fn(**overrides)
