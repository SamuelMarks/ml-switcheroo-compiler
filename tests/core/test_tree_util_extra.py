"""Test tree util extra."""

from ml_switcheroo_compiler.tree_util import tree_flatten


def test_tree_def_hash_eq() -> None:
    """Test tree def hash and eq."""
    flat1, tree_def1 = tree_flatten({"a": 1, "b": [2, 3]})
    flat2, tree_def2 = tree_flatten({"a": 1, "b": [2, 3]})
    flat3, tree_def3 = tree_flatten({"a": 1, "b": [2, 4]})  # same structure
    flat4, tree_def4 = tree_flatten({"a": 1, "b": {"c": 3}})

    assert hash(tree_def1) == hash(tree_def2)
    assert tree_def1 == tree_def2
    assert tree_def1 == tree_def3
    assert tree_def1 != tree_def4
    assert tree_def1 != "not a treedef"
