"""Core abstractions and logic definitions for test_ir_sharding.py."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ir.core import IRNode
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    pass


def test_ir_node_sharding() -> None:
    """Test the ir node sharding behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    node = IRNode("node_1", "Add")
    ensure_mutable(node).sharding = ensure_eager_tensor("shard_info")
    assert ensure_eager_tensor(node).sharding is not None
    assert ensure_eager_tensor(node.sharding) == ensure_eager_tensor("shard_info")
