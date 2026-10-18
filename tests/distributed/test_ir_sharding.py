"""Core abstractions and logic definitions for test_ir_sharding.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ir.core import IRNode

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_ir_node_sharding() -> None:
    """Test the ir node sharding behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    node = IRNode("node_1", "Add")
    setattr(node, "sharding", "shard_info")
    assert hasattr(node, "sharding")
    assert cast("EagerTensorProtocol", node.sharding) == cast("EagerTensorProtocol", "shard_info")
