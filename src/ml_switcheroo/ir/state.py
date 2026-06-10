"""State Mutation & Aliasing Representation for IR."""

import uuid

from ml_switcheroo.ir.core import IRNode


def create_read_variable(variable_name: str, shape: tuple, dtype: str) -> IRNode:
    """Create a ReadVariable node.

    Args:
        variable_name: The name of the variable to read.
        shape: The shape of the variable.
        dtype: The data type of the variable.

    Returns:
        The created IRNode.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="ReadVariable",
        inputs=[],
        attributes={"variable_name": variable_name},
        shape_metadata=shape,
    )


def create_assign_variable(variable_name: str, value_id: str, shape: tuple) -> IRNode:
    """Create an AssignVariable node.

    Args:
        variable_name: The name of the variable to assign to.
        value_id: The id of the node producing the new value.
        shape: The shape of the value.

    Returns:
        The created IRNode.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="AssignVariable",
        inputs=[value_id],
        attributes={"variable_name": variable_name},
        shape_metadata=shape,
    )


def create_scatter_update(
    tensor_id: str, indices_id: str, updates_id: str, shape: tuple
) -> IRNode:
    """Create a ScatterUpdate node.

    Args:
        tensor_id: The id of the tensor to update.
        indices_id: The id of the indices tensor.
        updates_id: The id of the updates tensor.
        shape: The output shape.

    Returns:
        The created IRNode.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="ScatterUpdate",
        inputs=[tensor_id, indices_id, updates_id],
        shape_metadata=shape,
    )
