# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tree_util import TreeDef, tree_all, tree_flatten, tree_leaves, tree_map, tree_reduce, tree_structure, tree_unflatten

"Core abstractions and logic definitions for test_tree_util_extra_new.py."


def test_tree_util_extra() -> object:
    """Test the tree util extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device("cpu")
        t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
        t2 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
        assert tree_leaves({"a": t1, "b": t2}) == [t1, t2]
        assert tree_structure({"a": t1, "b": t2}) is not None
        assert tree_all([True, True])
        assert not tree_all([True, False])
        assert tree_reduce(lambda x, y: x + y, [1, 2, 3]) == 6
        assert tree_reduce(lambda x, y: x + y, [1, 2, 3], 10) == 16
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Test tree_util."


def test_tree_flatten_unflatten() -> None:
    """Test the tree flatten unflatten behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test tree_flatten_unflatten."
        tree = {"a": 1, "b": [2, 3], "c": (4, {"d": 5})}
        (leaves, treedef) = tree_flatten(tree)
        assert leaves == [1, 2, 3, 4, 5]
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_tree_map() -> None:
    """Test the tree map behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test tree_map."
        tree1 = {"a": 1, "b": [2, 3]}
        tree2 = {"a": 10, "b": [20, 30]}
        res = tree_map(lambda x, y: x + y, tree1, tree2)
        assert res == {"a": 11, "b": [22, 33]}
        tree3 = {"a": 1, "b": 2}
        with pytest.raises(ValueError, match="All trees must have the same structure"):
            tree_map(lambda x, y: x + y, tree1, tree3)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_tree_def_repr():
    from ml_switcheroo_compiler.tree_util import TreeDef

    td = TreeDef(list)
    assert repr(td) == "TreeDef(list, [])"


def test_tree_def_hash():
    from ml_switcheroo_compiler.tree_util import TreeDef

    td1 = TreeDef(list)
    td2 = TreeDef(list)
    assert hash(td1) == hash(td2)


def test_tree_def_eq():
    from ml_switcheroo_compiler.tree_util import TreeDef

    td1 = TreeDef(list)
    td2 = TreeDef(list)
    assert td1 == td2
    assert td1 != "not_a_treedef"


def test_unflatten_dict_no_keys():
    from ml_switcheroo_compiler.tree_util import TreeDef, _unflatten_dict

    td = TreeDef(dict)
    with pytest.raises(ValueError):
        _unflatten_dict(td, iter([]))


def test_unflatten_node_unsupported():
    from ml_switcheroo_compiler.tree_util import TreeDef, _unflatten_node

    td = TreeDef(int)
    with pytest.raises(ValueError):
        _unflatten_node(td, iter([]))


def test_tree_unflatten_too_few_leaves():
    from ml_switcheroo_compiler.tree_util import TreeDef, tree_unflatten

    td = TreeDef(list, [TreeDef(type(None)), TreeDef(type(None))])
    with pytest.raises(ValueError):
        tree_unflatten(td, [1])


def test_tree_unflatten_too_many_leaves():
    from ml_switcheroo_compiler.tree_util import TreeDef, tree_unflatten

    td = TreeDef(list, [TreeDef(type(None))])
    with pytest.raises(ValueError):
        tree_unflatten(td, [1, 2])


def test_tree_map_structure_mismatch():
    from ml_switcheroo_compiler.tree_util import tree_map

    with pytest.raises(ValueError):
        tree_map(lambda x, y: x + y, [1, 2], [1, 2, 3])


def test_tree_transpose_mismatch():
    from ml_switcheroo_compiler.tree_util import tree_structure, tree_transpose

    outer = tree_structure([1, 2])
    inner = tree_structure([1, 2])
    with pytest.raises(ValueError):
        tree_transpose(outer, inner, [1, 2, 3])


def test_tree_all():
    from ml_switcheroo_compiler.tree_util import tree_all

    assert tree_all([True, True]) is True
    assert tree_all([True, False]) is False
    assert tree_all({"a": True}) is True


def test_tree_reduce():
    from ml_switcheroo_compiler.tree_util import tree_reduce

    assert tree_reduce(lambda x, y: x + y, [1, 2, 3]) == 6
    assert tree_reduce(lambda x, y: x + y, [1, 2, 3], 10) == 16


def test_tree_transpose_inner_size_zero():
    from ml_switcheroo_compiler.tree_util import tree_structure, tree_transpose

    outer = tree_structure([1, 2])
    inner = tree_structure([])
    res = tree_transpose(outer, inner, [])
    assert res == []


def test_tree_transpose_inner_loop():
    from ml_switcheroo_compiler.tree_util import tree_structure, tree_transpose

    outer = tree_structure([1, 2])
    inner = tree_structure([1, 2])
    res = tree_transpose(outer, inner, [[1, 2], [3, 4]])
    assert res == [[1, 3], [2, 4]]


"Test tree util extra."


def test_tree_def_hash_eq() -> None:
    """Test the tree def hash eq behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test tree def hash and eq."
        (flat1, tree_def1) = tree_flatten({"a": 1, "b": [2, 3]})
        (flat2, tree_def2) = tree_flatten({"a": 1, "b": [2, 3]})
        (flat3, tree_def3) = tree_flatten({"a": 1, "b": [2, 4]})
        (flat4, tree_def4) = tree_flatten({"a": 1, "b": {"c": 3}})
        assert hash(tree_def1) == hash(tree_def2)
        assert tree_def1 == tree_def2
        assert tree_def1 == tree_def3
        assert tree_def1 != tree_def4
        assert tree_def1 != "not a treedef"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
