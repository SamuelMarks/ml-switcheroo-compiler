"""Module core.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Unified Intermediate Representation (IR) Schema."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ml_switcheroo_ir import (
    AttributeValue,
    LogicalGraph,
    LogicalNode,
    PartitionSpec,
)
from ml_switcheroo_ir import (
    LogicalGraph as IRGraph,
)
from ml_switcheroo_ir import (
    LogicalNode as IRNode,
)
from ml_switcheroo_ir.types import DType, TensorSpec

# Re-export TensorSpec from ml_switcheroo_ir.types
TensorSpec = TensorSpec

__all__ = [
    "DType",
    "IRBlock",
    "IRGraph",
    "IRNode",
    "LogicalGraph",
    "LogicalNode",
    "NoTangent",
    "TensorSpec",
    "ZeroTangent",
    "clone_logical_node",
]


def clone_logical_node(
    node: LogicalNode,
    **kwargs: (str | int | float | bool | list[str] | dict[str, AttributeValue] | dict[str, LogicalGraph] | list[TensorSpec] | Sequence[int | str] | PartitionSpec | DType | None),
) -> LogicalNode:
    """Clones a LogicalNode, allowing overrides via kwargs.

    Args:
        node (LogicalNode): The node parameter.
        **kwargs: Keyword overrides.

    Returns:
        LogicalNode: The cloned LogicalNode.
    """
    attributes = dict(node.attributes)
    inputs = list(node.inputs)
    outputs = list(node.outputs) if getattr(node, "outputs", None) is not None else None
    output_specs = list(node.output_specs) if getattr(node, "output_specs", None) is not None else []
    subgraphs = dict(node.subgraphs) if getattr(node, "subgraphs", None) is not None else {}

    clone_kwargs: dict[
        str,
        (str | int | float | bool | list[str] | dict[str, AttributeValue] | dict[str, LogicalGraph] | list[TensorSpec] | Sequence[int | str] | PartitionSpec | DType | None),
    ] = {
        "id": node.id,
        "op_type": node.op_type,
        "domain": node.domain,
        "version": node.version,
        "attributes": attributes,
        "inputs": inputs,
        "outputs": outputs,
        "shape_metadata": node.shape_metadata,
        "source_ast_ref": node.source_ast_ref,
        "sharding": node.sharding,
        "dtype": getattr(node, "dtype", None),
        "output_specs": output_specs,
        "subgraphs": subgraphs,
        "device": getattr(node, "device", None),
        "stream": getattr(node, "stream", None),
    }
    clone_kwargs.update(kwargs)
    return LogicalNode(**clone_kwargs)


class IRBlock(LogicalGraph):
    """Deprecated: Legacy container for nested control flow scopes; use LogicalGraph instead.

    Attributes:
        id: Unique identifier for the block (maps to graph name).
        nodes: The nodes dictionary in this block.
        inputs: List of input variable names from the outer scope.
        outputs: List of output variable names.
    """

    def __init__(
        self,
        id: str,
        nodes: list[LogicalNode] | dict[str, LogicalNode] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
    ) -> None:
        """Initialize deprecated IRBlock.

        Args:
            id (str): Unique identifier for the block.
            nodes (list | dict | None): Nodes contained in this block.
            inputs (list | None): Input node IDs.
            outputs (list | None): Output node IDs.
        """
        import warnings

        warnings.warn("IRBlock is deprecated, use LogicalGraph instead.", DeprecationWarning, stacklevel=2)
        super().__init__(name=id, nodes=nodes, outputs=outputs, inputs=inputs)
        self.id = id


class ZeroTangent(LogicalNode):
    """Represents a mathematically zero tangent (e.g. gradient wrt integer or unconnected)."""

    def __init__(
        self,
        id: str,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | None = None,
        **kwargs: str | int | float | bool | list[str] | dict[str, AttributeValue] | None,
    ) -> None:
        """Initialize ZeroTangent node.

        Args:
            id (str): The node id.
            shape_metadata: Shape metadata.
            **kwargs: Additional kwargs.
        """
        super().__init__(id=id, op_type="ZeroTangent", shape_metadata=shape_metadata, **kwargs)


class NoTangent(LogicalNode):
    """Represents a structurally missing or non-differentiable tangent path."""

    def __init__(
        self,
        id: str,
        **kwargs: str | int | float | bool | list[str] | dict[str, AttributeValue] | None,
    ) -> None:
        """Initialize NoTangent node.

        Args:
            id (str): The node id.
            **kwargs: Additional kwargs.
        """
        super().__init__(id=id, op_type="NoTangent", **kwargs)
