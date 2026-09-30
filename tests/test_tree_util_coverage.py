"""Tests for test_tree_util_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo_compiler.tree_util as tree_util_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, emit_ir_node


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy object with shape.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize dummy object without shape."""
        self.val: int = 42


def test_tree_util_and_ops_base_confirmation() -> None:
    """Confirm tree_util and ops.base full integration coverage."""
    tree = {"a": [1, 2, (3, 4)], "b": 5}
    leaves, structure = tree_util_mod.tree_flatten(tree)
    assert leaves == [1, 2, 3, 4, 5]
    reconstructed = tree_util_mod.tree_unflatten(structure, leaves)
    assert reconstructed == tree

    mapped = tree_util_mod.tree_map(lambda x: x * 2, tree)
    assert mapped == {"a": [2, 4, (6, 8)], "b": 10}

    assert tree_util_mod.tree_leaves({"a": 1, "b": 2}) == [1, 2]
    assert tree_util_mod.tree_structure({"a": 1, "b": 2}) is not None
    assert tree_util_mod.tree_all([True, True])
    assert not tree_util_mod.tree_all([True, False])
    assert tree_util_mod.tree_reduce(lambda x, y: x + y, [1, 2, 3]) == 6
    assert tree_util_mod.tree_reduce(lambda x, y: x + y, [1, 2, 3], 10) == 16

    td1 = tree_util_mod.TreeDef(list, [tree_util_mod.TreeDef(type(None))])
    td2 = tree_util_mod.TreeDef(list, [tree_util_mod.TreeDef(type(None))])
    assert td1 == td2
    assert td1 != "not_a_treedef"
    assert hash(td1) == hash(td2)
    assert repr(td1) == "TreeDef(list, [TreeDef(NoneType, [])])"

    with pytest.raises(ValueError, match="All trees must have the same structure"):
        tree_util_mod.tree_map(lambda x, y: x + y, [1, 2], [1, 2, 3])

    with pytest.raises(ValueError, match="Too few leaves"):
        tree_util_mod.tree_unflatten(td1, [])

    with pytest.raises(ValueError, match="Too many leaves"):
        tree_util_mod.tree_unflatten(td1, [1, 2])

    with pytest.raises(ValueError, match="Unsupported treedef node_type"):
        tree_util_mod._unflatten_node(tree_util_mod.TreeDef(int), iter([]))

    with pytest.raises(ValueError, match="Dict treedef must have keys"):
        tree_util_mod._unflatten_dict(tree_util_mod.TreeDef(dict), iter([]))

    outer = tree_util_mod.tree_structure([[1, 2], [3, 4]])
    inner = tree_util_mod.tree_structure([1, 2])
    with pytest.raises(ValueError):
        tree_util_mod.tree_transpose(outer, inner, [1, 2])

    dummy_op = OpDef()
    assert dummy_op.infer_shape() == ()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_be:
        backend = MagicMock()
        backend.execute_op.return_value = 99
        mock_be.return_value = backend
        assert dummy_op.eager_eval(1, 2) == 99

    class DummyGraph:
        """Mock IR graph container."""

        def __init__(self) -> None:
            """Initialize node dictionary."""
            self.nodes: dict[str, IRNode] = {}

    graph = DummyGraph()
    nid = emit_ir_node(graph, "CustomTestOp", ["input_0"])
    assert nid in graph.nodes

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node") as mock_add_node:
        nid_none = emit_ir_node(None, "CustomTestOpNoGraph", ["input_1"])
        assert nid_none is not None
        mock_add_node.assert_called_once()

    @dispatch_eager("DummyEagerOp")
    def dummy_func(x: int) -> int:
        """Dummy function for eager dispatch testing.

        Args:
            x (int): Input integer.

        Returns:
            int: Unmodified integer.
        """
        return x

    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        assert dummy_func(42) == 42
    finally:
        config.eager_mode = original_eager
