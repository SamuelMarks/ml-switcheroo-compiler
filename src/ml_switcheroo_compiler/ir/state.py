"""Canonical IR state operation constants and node factory constructors (State Layer 1).

Layer Responsibilities across the State Subsystem:
    1. ``ir.state``: Canonical IR op identifiers and stateless ``IRNode`` constructors.
    2. ``core.state_manager``: Pure graph-level ``Assign`` environment tracking and functionalization.
    3. ``tracing.state``: Thread/context-active graph builder state during symbolic tracing.
    4. ``ops.state``: Registered ``OpDef`` schemas (``ReadVariable``, ``AssignVariable``).
"""

from __future__ import annotations

import uuid

from ml_switcheroo_ir import IRNode, NodeId, OpName

from ml_switcheroo_compiler.ir.core import ShapeLike

READ_VARIABLE_OP: OpName = OpName("ReadVariable")
ASSIGN_VARIABLE_OP: OpName = OpName("AssignVariable")
SCATTER_UPDATE_OP: OpName = OpName("ScatterUpdate")
ASSIGN_OP: OpName = OpName("Assign")


def create_read_variable(variable_name: str, shape: ShapeLike, dtype: str) -> IRNode:
    """Create a ``ReadVariable`` IR node reading the current value of a named state variable.

    Args:
        variable_name (str): Name of the state variable to read.
        shape (ShapeLike): Output tensor shape metadata.
        dtype (str): Output tensor data type identifier.

    Returns:
        IRNode: Constructed ``ReadVariable`` logical node.
    """
    return IRNode(
        id=NodeId(str(uuid.uuid4())),
        op_type=READ_VARIABLE_OP,
        inputs=[],
        attributes={"variable_name": variable_name, "dtype": dtype},
        dtype=dtype,
        shape_metadata=shape,
    )


def create_assign_variable(variable_name: str, value_id: NodeId | str, shape: ShapeLike) -> IRNode:
    """Create an ``AssignVariable`` IR node mutating a named state variable to ``value_id``.

    Args:
        variable_name (str): Name of the state variable to update.
        value_id (NodeId | str): Node ID of the value to assign.
        shape (ShapeLike): Output tensor shape metadata.

    Returns:
        IRNode: Constructed ``AssignVariable`` logical node.
    """
    return IRNode(
        id=NodeId(str(uuid.uuid4())),
        op_type=ASSIGN_VARIABLE_OP,
        inputs=[str(value_id)],
        attributes={"variable_name": variable_name},
        shape_metadata=shape,
    )


def create_scatter_update(
    tensor_id: NodeId | str,
    indices_id: NodeId | str,
    updates_id: NodeId | str,
    shape: ShapeLike,
) -> IRNode:
    """Create a ``ScatterUpdate`` IR node updating indexed slices of a tensor.

    Args:
        tensor_id (NodeId | str): Node ID of the base tensor.
        indices_id (NodeId | str): Node ID of the scatter index tensor.
        updates_id (NodeId | str): Node ID of the replacement values tensor.
        shape (ShapeLike): Output tensor shape metadata.

    Returns:
        IRNode: Constructed ``ScatterUpdate`` logical node.
    """
    return IRNode(
        id=NodeId(str(uuid.uuid4())),
        op_type=SCATTER_UPDATE_OP,
        inputs=[str(tensor_id), str(indices_id), str(updates_id)],
        shape_metadata=shape,
    )


__all__ = [
    "ASSIGN_OP",
    "ASSIGN_VARIABLE_OP",
    "READ_VARIABLE_OP",
    "SCATTER_UPDATE_OP",
    "create_assign_variable",
    "create_read_variable",
    "create_scatter_update",
]
