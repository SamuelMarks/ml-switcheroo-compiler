"""Core abstractions and logic definitions for test_ir_sharding.py."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ir.core import IRNode

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_ir_node_sharding() -> None:
    """Test the ir node sharding behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    node = IRNode("node_1", "Add")
    object.__setattr__(node, "sharding", "shard_info")
    assert hasattr(node, "sharding")
    assert typing.cast("EagerTensorProtocol", node.sharding) == typing.cast("EagerTensorProtocol", "shard_info")
