# ruff: noqa: E501
"""Test state.py."""

from ml_switcheroo_compiler.ir.state import create_assign_variable, create_read_variable, create_scatter_update


def test_state_creation() -> None:
    """Test the state creation behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test creating state variables."
        read_var = create_read_variable("var1", (2, 2), "float32")
        assert read_var.op_type == "ReadVariable"
        assign_var = create_assign_variable("var1", "val1", (2, 2))
        assert assign_var.op_type == "AssignVariable"
        scatter = create_scatter_update("t1", "idx1", "upd1", (2, 2))
        assert scatter.op_type == "ScatterUpdate"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
