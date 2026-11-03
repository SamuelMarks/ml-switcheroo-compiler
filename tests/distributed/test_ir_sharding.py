"""Core abstractions and logic definitions for test_ir_sharding.py."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import _has_attr, _obj_set_attr
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
    _obj_set_attr(node, "sharding", "shard_info")
    assert _has_attr(node, "sharding")
    assert as_tensor(node.sharding) == as_tensor("shard_info")
