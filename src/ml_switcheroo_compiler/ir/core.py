"""Unified Intermediate Representation (IR) Schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ml_switcheroo_ir import LogicalGraph, LogicalNode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ml_switcheroo_compiler.core.dtype import DType


@dataclass
class IRNode(LogicalNode):
    """Extended LogicalNode for ml_switcheroo_compiler compiler internal IR."""

    stream: str | None = None
    device: str | None = None


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
