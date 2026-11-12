"""Broadcast Explicitizer Pass."""

from __future__ import annotations

import uuid
from collections.abc import (
    Sequence,  # noqa: F401
)

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.constants import BINARY_OP_ARITY
from ml_switcheroo_compiler.core.shape import SymbolicDim, broadcast_shapes
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass


def _inject_broadcast_node(graph: IRGraph, input_id: str, target_shape: tuple[int, ...]) -> str:
    """Execute inject broadcast node for compiler processing.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        input_id (str): Input `input_id` argument (input id).
        target_shape (tuple): Input `target_shape` argument (target shape).

    Returns:
        str: Computed `str` output.
    """
    new_id = f"broadcast_{uuid.uuid4().hex[:6]}"
    new_node = LogicalNode(
        id=new_id,
        op_type="BroadcastTo",
        inputs=[input_id],
        shape_metadata=target_shape,
        attributes={"shape": target_shape},
    )
    graph.nodes[new_id] = new_node
    return new_id


def _to_int_shape(seq: Sequence[int | SymbolicDim | None]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[int | SymbolicDim | None]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.

    Raises:
        TypeError: If any dimension is not an integer.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


def _needs_broadcast(
    shape1: Sequence[int | SymbolicDim | None] | None,
    shape2: Sequence[int | SymbolicDim | None] | None,
) -> tuple[int, ...] | None:
    """Execute needs broadcast for compiler processing.

    Args:
        shape1 (Sequence[int | SymbolicDim | None] | None): Input `shape1` argument (shape1).
        shape2 (Sequence[int | SymbolicDim | None] | None): Input `shape2` argument (shape2).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if shape1 is None or shape2 is None or shape1 == shape2:
        return None
    try:
        return broadcast_shapes(_to_int_shape(shape1), _to_int_shape(shape2))
    except Exception:
        return None


def _process_broadcast_node(graph: IRGraph, node: LogicalNode) -> bool:
    """Process node.

    Args:
        graph (IRGraph): Graph.
        node (LogicalNode): Node.

    Returns:
        bool: Computed `bool` output.
    """
    try:
        get_op(node.op_type)
    except KeyError:
        return False

    if len(node.inputs) != BINARY_OP_ARITY:
        return False

    in1, in2 = node.inputs
    shape1 = graph.nodes[in1].shape_metadata
    shape2 = graph.nodes[in2].shape_metadata

    target_shape = _needs_broadcast(shape1, shape2)
    if target_shape is None:
        return False

    modified = False
    if shape1 != target_shape:
        node.inputs[0] = _inject_broadcast_node(graph, in1, target_shape)
        modified = True

    if shape2 != target_shape:
        node.inputs[1] = _inject_broadcast_node(graph, in2, target_shape)
        modified = True

    return modified


def broadcast_explicitizer_pass(graph: IRGraph) -> bool:
    """In-place pass to explicitly inject BroadcastTo nodes.

    Args:
        graph (IRGraph): The graph parameter for the operation.

    Returns:
        bool: A boolean indicating the result of the check.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    shape_inference_pass(graph)

    for node in sorted_nodes:
        if _process_broadcast_node(graph, node):
            modified = True

    return modified
