"""State Mutation & Aliasing Representation for IR."""

from __future__ import annotations

import uuid

from ml_switcheroo_compiler.ir.core import IRNode, ShapeLike


def create_read_variable(variable_name: str, shape: ShapeLike, dtype: str) -> IRNode:
    """Create a `ReadVariable` IR node reading the current value of a named state variable.

    Args:
        variable_name (str): Name of the state variable to read.
        shape (ShapeLike): Output tensor shape metadata.
        dtype (str): Output tensor data type identifier.

    Returns:
        IRNode: Constructed `ReadVariable` logical node.
    """
    _ = dtype
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="ReadVariable",
        inputs=[],
        attributes={"variable_name": variable_name},
        shape_metadata=shape,
    )


def create_assign_variable(variable_name: str, value_id: str, shape: ShapeLike) -> IRNode:
    """Create an `AssignVariable` IR node mutating a named state variable to `value_id`.

    Args:
        variable_name (str): Name of the state variable to update.
        value_id (str): Node ID of the value to assign.
        shape (ShapeLike): Output tensor shape metadata.

    Returns:
        IRNode: Constructed `AssignVariable` logical node.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="AssignVariable",
        inputs=[value_id],
        attributes={"variable_name": variable_name},
        shape_metadata=shape,
    )


def create_scatter_update(
    tensor_id: str,
    indices_id: str,
    updates_id: str,
    shape: ShapeLike,
) -> IRNode:
    """Create a `ScatterUpdate` IR node updating indexed slices of a tensor.

    Args:
        tensor_id (str): Node ID of the base tensor.
        indices_id (str): Node ID of the scatter index tensor.
        updates_id (str): Node ID of the replacement values tensor.
        shape (ShapeLike): Output tensor shape metadata.

    Returns:
        IRNode: Constructed `ScatterUpdate` logical node.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="ScatterUpdate",
        inputs=[tensor_id, indices_id, updates_id],
        shape_metadata=shape,
    )
