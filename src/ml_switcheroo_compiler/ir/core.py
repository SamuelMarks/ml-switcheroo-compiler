"""Unified Intermediate Representation (IR) Schema."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.dtype import DType


def clone_logical_node(node: LogicalNode, **kwargs: object) -> LogicalNode:
    """Clones a LogicalNode, allowing overrides via kwargs."""
    attributes = dict(node.attributes)
    inputs = list(node.inputs)

    clone_kwargs = {
        "id": node.id,
        "op_type": node.op_type,
        "domain": node.domain,
        "version": node.version,
        "attributes": attributes,
        "inputs": inputs,
        "shape_metadata": node.shape_metadata,
        "source_ast_ref": node.source_ast_ref,
        "sharding": node.sharding,
    }
    clone_kwargs.update(kwargs)
    return LogicalNode(**clone_kwargs)


@dataclass
class IRNode(LogicalNode):
    """Extended LogicalNode for ml_switcheroo_compiler compiler internal IR."""

    stream: str | None = None
    device: str | None = None

    @property
    def is_dynamic_shape(self) -> bool:
        """Return whether the node's shape metadata is dynamic.

        Returns:
            bool: True if the shape is dynamic or unknown.
        """
        if self.shape_metadata is None:
            return False
        if not isinstance(self.shape_metadata, (tuple, list)):
            return False
        return any(not isinstance(dim, int) for dim in self.shape_metadata)

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return the static shape, raising an error if it's dynamic.

        Returns:
            tuple[int, ...]: The static shape.

        Raises:
            ValueError: If the shape is dynamic or not available.
        """
        if self.shape_metadata is None or not isinstance(self.shape_metadata, (tuple, list)):
            msg = "Shape metadata is not available or not a sequence."
            raise ValueError(msg)
        if self.is_dynamic_shape:
            msg = f"Cannot get static shape from dynamic node shape: {self.shape_metadata}"
            raise ValueError(msg)
        return tuple(int(dim) for dim in self.shape_metadata)

    @property
    def rank(self) -> int:
        """Return the number of dimensions of the node's output.

        Returns:
            int: The rank of the node.
        """
        if self.shape_metadata is None or not isinstance(self.shape_metadata, (tuple, list)):
            return 0
        return len(self.shape_metadata)


# Re-export base IR classes
IRGraph = LogicalGraph


@dataclass
class TensorSpec:
    """Specification of a tensor's properties in the IR.

    Attributes:
        shape: The shape of the tensor. Can contain strings for dynamic dims
        dtype: The data type of the tensor
        sparsity: Optional sparsity pattern metadata
    """

    shape: Sequence[int | str]
    dtype: DType
    sparsity: dict[str, Any] | None = None

    @property
    def is_dynamic(self) -> bool:
        """Return whether any dimension in the shape is dynamic.

        Returns:
            bool: True if any dimension is not a static integer.
        """
        return any(not isinstance(dim, int) for dim in self.shape)

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return the static shape as a tuple of integers.

        Returns:
            tuple[int, ...]: The static shape.

        Raises:
            ValueError: If the shape contains dynamic dimensions.
        """
        if self.is_dynamic:
            msg = f"Cannot get static shape from dynamic tensor shape: {self.shape}"
            raise ValueError(msg)
        return tuple(int(dim) for dim in self.shape)  # type: ignore

    @property
    def rank(self) -> int:
        """Return the number of dimensions (rank) of the tensor.

        Returns:
            int: The rank of the tensor.
        """
        return len(self.shape)


@dataclass
class IRBlock:
    """Represents nested scopes for control flow like cond and while_loop.

    Attributes:
        id: Unique identifier for the block
        nodes: The list of nodes in this block
        inputs: List of input variable names from the outer scope
        outputs: List of output variable names
    """

    id: str
    nodes: list[IRNode] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
