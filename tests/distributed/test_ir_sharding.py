"""Core abstractions and logic definitions for test_ir_sharding.py."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import IRNode

if typing.TYPE_CHECKING:
    pass


def test_ir_node_sharding() -> None:
    """Test the ir node sharding behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    node = IRNode("node_1", "Add")
    node.sharding = "shard_info"
    assert node.sharding is not None
    assert node.sharding == "shard_info"
