"""Unified Intermediate Representation (IR) Schema."""

from typing import Dict, List, Any, Optional, Sequence, Union
from dataclasses import dataclass, field
from ml_switcheroo_ir import LogicalNode, LogicalGraph
from ml_switcheroo.core.dtype import DType

# Re-export base IR classes
IRNode = LogicalNode
IRGraph = LogicalGraph


@dataclass
class TensorSpec:
    """Specification of a tensor's properties in the IR.

    Attributes:
        shape: The shape of the tensor. Can contain strings for dynamic dims.
        dtype: The data type of the tensor.
        sparsity: Optional sparsity pattern metadata.
    """

    shape: Sequence[Union[int, str]]
    dtype: DType
    sparsity: Optional[Dict[str, Any]] = None


@dataclass
class IRBlock:
    """Represents nested scopes for control flow like cond and while_loop.

    Attributes:
        id: Unique identifier for the block.
        nodes: The list of nodes in this block.
        inputs: List of input variable names from the outer scope.
        outputs: List of output variable names.
    """

    id: str
    nodes: List[IRNode] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
