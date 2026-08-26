# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module state.py."""

"""State Mutation & Aliasing Representation for IR."""

import uuid

from ml_switcheroo_compiler.ir.core import IRNode


def create_read_variable(variable_name: str, shape, dtype: str) -> IRNode:
    """Create a ReadVariable node.

    Args:
        variable_name (str): The variable_name parameter.
        shape (tuple): The shape parameter.
        dtype (str): The dtype parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="ReadVariable",
        inputs=[],
        attributes={"variable_name": variable_name},
        shape_metadata=shape,
    )


def create_assign_variable(variable_name: str, value_id: str, shape) -> IRNode:
    """Create an AssignVariable node.

    Args:
        variable_name (str): The variable_name parameter.
        value_id (str): The value_id parameter.
        shape (tuple): The shape parameter.

    Returns:
        IRNode: Result.
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
    shape,
) -> IRNode:
    """Create a ScatterUpdate node.

    Args:
        tensor_id (str): The tensor_id parameter.
        indices_id (str): The indices_id parameter.
        updates_id (str): The updates_id parameter.
        shape (tuple): The shape parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(
        id=str(uuid.uuid4()),
        op_type="ScatterUpdate",
        inputs=[tensor_id, indices_id, updates_id],
        shape_metadata=shape,
    )
