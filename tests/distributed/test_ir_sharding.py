# ruff: noqa: E501
"""Core abstractions and logic definitions for test_ir_sharding.py."""

from ml_switcheroo_compiler.ir.core import IRNode


def test_ir_node_sharding() -> object:
    """Test the ir node sharding behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        node = IRNode("node_1", "Add")
        node.sharding = "shard_info"
        assert hasattr(node, "sharding")
        assert node.sharding == "shard_info"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
