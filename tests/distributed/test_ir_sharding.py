"""Core abstractions and logic definitions for test_ir_sharding.py."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ir.core import IRNode
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_ir_node_sharding() -> None:
    """Test the ir node sharding behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    node = IRNode("node_1", "Add")
    object.__setattr__(node, "sharding", "shard_info")
    assert hasattr(node, "sharding")
    assert as_tensor(node.sharding) == as_tensor("shard_info")
