"""Test tree_util."""

import pytest

from ml_switcheroo_compiler.tree_util import TreeDef, tree_flatten, tree_map, tree_unflatten


def test_tree_flatten_unflatten() -> None:
    """Test tree_flatten_unflatten."""
    tree = {"a": 1, "b": [2, 3], "c": (4, {"d": 5})}
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4, 5]

    # Test representation and equality
    assert repr(treedef).startswith("TreeDef")
    assert treedef == treedef
    assert treedef != TreeDef(list)
    assert treedef != "Not a TreeDef"

    tree_reconstructed = tree_unflatten(treedef, leaves)
    assert tree == tree_reconstructed

    with pytest.raises(ValueError, match="Too few leaves"):
        tree_unflatten(treedef, [1, 2, 3, 4])

    with pytest.raises(ValueError, match="Too many leaves"):
        tree_unflatten(treedef, [1, 2, 3, 4, 5, 6])

    with pytest.raises(ValueError, match="Unsupported treedef node_type"):
        tree_unflatten(TreeDef(set), [])

    with pytest.raises(ValueError, match="Dict treedef must have keys"):
        tree_unflatten(TreeDef(dict), [])


def test_tree_map() -> None:
    """Test tree_map."""
    tree1 = {"a": 1, "b": [2, 3]}
    tree2 = {"a": 10, "b": [20, 30]}

    res = tree_map(lambda x, y: x + y, tree1, tree2)
    assert res == {"a": 11, "b": [22, 33]}

    tree3 = {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="All trees must have the same structure"):
        tree_map(lambda x, y: x + y, tree1, tree3)
