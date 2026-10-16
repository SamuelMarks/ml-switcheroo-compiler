"""Unified Intermediate Representation (IR) Schema."""

from __future__ import annotations

# ruff: noqa: F401
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Union, cast

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
from ml_switcheroo_ir.types import TensorSpec

from ml_switcheroo_compiler.core.dtype import DType

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


LogicalNodeOverrideValue = Union[
    str,
    int,
    dict[str, AttributeValue],
    list[str],
    Sequence[Union[int, str]],
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
    attributes = overrides.get("attributes")
    inputs = overrides.get("inputs")
    outputs = overrides.get("outputs")
    output_specs = overrides.get("output_specs")
    subgraphs = overrides.get("subgraphs")

    node_attrs: dict[str, AttributeValue] = dict(node.attributes) if not isinstance(attributes, dict) else dict(cast("dict[str, AttributeValue]", attributes))
    node_inputs: list[str] = list(node.inputs) if not isinstance(inputs, list) else list(cast("list[str]", inputs))
    raw_outputs = getattr(node, "outputs", None)
    node_outputs: list[str] | None = (list(raw_outputs) if raw_outputs is not None else None) if not isinstance(outputs, list) else list(cast("list[str]", outputs))
    raw_specs = getattr(node, "output_specs", None)
    node_specs: list[TensorSpec] = (list(raw_specs) if raw_specs is not None else []) if not isinstance(output_specs, list) else [x for x in output_specs]
    raw_subgraphs = getattr(node, "subgraphs", None)
    node_subgraphs: dict[str, LogicalGraph] = (dict(raw_subgraphs) if raw_subgraphs is not None else {}) if not isinstance(subgraphs, dict) else dict(cast("dict[str, LogicalGraph]", subgraphs))

    ver_override = overrides.get("version")
    node_version: int = int(ver_override) if isinstance(ver_override, (int, str)) else node.version
    shape_override = overrides.get("shape_metadata")
    node_shape: Sequence[int | str] | None = cast("Sequence[int | str]", shape_override) if isinstance(shape_override, (list, tuple)) else node.shape_metadata
    sharding_override = overrides.get("sharding")
    node_sharding: PartitionSpec | None = sharding_override if isinstance(sharding_override, PartitionSpec) else node.sharding
    dtype_override = overrides.get("dtype")
    node_dtype: DType | str | None = dtype_override if isinstance(dtype_override, (DType, str)) else getattr(node, "dtype", None)
    node_id = node.id if overrides.get("id") is None else str(overrides["id"])
    node_op = node.op_type if overrides.get("op_type") is None else str(overrides["op_type"])
    node_domain = node.domain if overrides.get("domain") is None else str(overrides["domain"])
    node_ast_ref = node.source_ast_ref if overrides.get("source_ast_ref") is None else str(overrides["source_ast_ref"])
    node_device = getattr(node, "device", None) if overrides.get("device") is None else str(overrides["device"])
    node_stream = getattr(node, "stream", None) if overrides.get("stream") is None else str(overrides["stream"])

    return cast("Callable[..., LogicalNode]", LogicalNode)(
        id=node_id,
        op_type=node_op,
        domain=node_domain,
        version=node_version,
        attributes=node_attrs,
        inputs=node_inputs,
        outputs=node_outputs,
        shape_metadata=node_shape,
        source_ast_ref=node_ast_ref,
        sharding=node_sharding,
        dtype=node_dtype,
        output_specs=node_specs,
        subgraphs=node_subgraphs,
        device=node_device,
        stream=node_stream,
    )


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
        inputs: list[str] | None = None,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize ZeroTangent node.

        Args:
            id (str): The node id.
            shape_metadata: Shape metadata.
            inputs (list[str] | None): Optional input node IDs.
            attributes (dict[str, AttributeValue] | None): Optional node attributes.
        """
        super().__init__(
            id=id,
            op_type="ZeroTangent",
            shape_metadata=shape_metadata,
            inputs=inputs if inputs is not None else [],
            attributes=attributes if attributes is not None else {},
        )


class NoTangent(LogicalNode):
    """Represents a structurally missing or non-differentiable tangent path."""

    def __init__(
        self,
        id: str,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | None = None,
        inputs: list[str] | None = None,
        attributes: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize NoTangent node.

        Args:
            id (str): The node id.
            shape_metadata: Shape metadata.
            inputs (list[str] | None): Optional input node IDs.
            attributes (dict[str, AttributeValue] | None): Optional node attributes.
        """
        super().__init__(
            id=id,
            op_type="NoTangent",
            shape_metadata=shape_metadata,
            inputs=inputs if inputs is not None else [],
            attributes=attributes if attributes is not None else {},
        )
