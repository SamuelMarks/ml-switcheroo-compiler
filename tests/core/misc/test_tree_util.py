"""Tests for test_tree_util."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tree_util import TreeDef, tree_all, tree_flatten, tree_leaves, tree_map, tree_reduce, tree_structure, tree_unflatten
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    pass


def test_tree_util_extra() -> None:
    """Test the tree util extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    device = Device("cpu")
    t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    t2 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    assert tree_leaves({"a": t1, "b": t2}) == [t1, t2]
    assert tree_structure({"a": t1, "b": t2}) is not None
    assert tree_all([True, True])
    assert not tree_all([True, False])
    assert as_callable(tree_reduce)(lambda x, y: x + y, [1, 2, 3]) == 6
    assert as_callable(tree_reduce)(lambda x, y: x + y, [1, 2, 3], 10) == 16


def test_tree_flatten_unflatten() -> None:
    """Test the tree flatten unflatten behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    tree = {"a": 1, "b": [2, 3], "c": (4, {"d": 5})}
    (leaves, treedef) = as_callable(tree_flatten)(tree)
    assert leaves == [1, 2, 3, 4, 5]
    assert repr(treedef).startswith("TreeDef")
    (_, treedef2) = as_callable(tree_flatten)({"a": 1, "b": [2, 3], "c": (4, {"d": 5})})
    assert treedef == treedef2
    assert hash(treedef) == hash(treedef2)
    (_, mutated_treedef) = as_callable(tree_flatten)({"a": 10, "b": [20, 30], "c": (40, {"d": 50})})
    assert treedef == mutated_treedef
    (_, different_treedef) = tree_flatten({"a": [1, 2], "b": [3, 4, 5]})
    assert treedef != different_treedef
    assert hash(treedef) != hash(different_treedef)
    assert treedef != TreeDef(list)
    assert treedef != "Not a TreeDef"
    tree_reconstructed = as_callable(tree_unflatten)(treedef, leaves)
    assert tree == tree_reconstructed
    with pytest.raises(ValueError, match="Too few leaves"):
        as_callable(tree_unflatten)(treedef, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="Too many leaves"):
        as_callable(tree_unflatten)(treedef, [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="Unsupported treedef node_type"):
        tree_unflatten(TreeDef(set), [])
    with pytest.raises(ValueError, match="Dict treedef must have keys"):
        tree_unflatten(TreeDef(dict), [])


def test_tree_map() -> None:
    """Test the tree map behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    tree1 = {"a": 1, "b": [2, 3]}
    tree2 = {"a": 10, "b": [20, 30]}
    res = as_callable(tree_map)(lambda x, y: x + y, tree1, tree2)
    assert res == {"a": 11, "b": [22, 33]}
    tree3 = {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="All trees must have the same structure"):
        as_callable(tree_map)(lambda x, y: x + y, tree1, tree3)


def test_tree_def_repr() -> None:
    """Test function test_tree_def_repr."""
    from ml_switcheroo_compiler.tree_util import TreeDef

    td = TreeDef(list)
    assert repr(td) == "TreeDef(list, [])"


def test_tree_def_hash() -> None:
    """Test function test_tree_def_hash."""
    from ml_switcheroo_compiler.tree_util import TreeDef

    td1 = TreeDef(list)
    td2 = TreeDef(list)
    assert hash(td1) == hash(td2)


def test_tree_def_eq() -> None:
    """Test function test_tree_def_eq."""
    from ml_switcheroo_compiler.tree_util import TreeDef

    td1 = TreeDef(list)
    td2 = TreeDef(list)
    assert td1 == td2
    assert as_tensor(td1) != as_tensor("not_a_treedef")


def test_unflatten_dict_no_keys() -> None:
    """Test function test_unflatten_dict_no_keys."""
    from ml_switcheroo_compiler.tree_util import TreeDef, _unflatten_dict

    td = TreeDef(dict)
    with pytest.raises(ValueError):
        _unflatten_dict(td, iter([]))


def test_unflatten_node_unsupported() -> None:
    """Test function test_unflatten_node_unsupported."""
    from ml_switcheroo_compiler.tree_util import TreeDef, _unflatten_node

    td = TreeDef(int)
    with pytest.raises(ValueError):
        _unflatten_node(td, iter([]))


def test_tree_unflatten_too_few_leaves() -> None:
    """Test function test_tree_unflatten_too_few_leaves."""
    from ml_switcheroo_compiler.tree_util import TreeDef, tree_unflatten

    td = TreeDef(list, [TreeDef(type(None)), TreeDef(type(None))])
    with pytest.raises(ValueError):
        tree_unflatten(td, [1])


def test_tree_unflatten_too_many_leaves() -> None:
    """Test function test_tree_unflatten_too_many_leaves."""
    from ml_switcheroo_compiler.tree_util import TreeDef, tree_unflatten

    td = TreeDef(list, [TreeDef(type(None))])
    with pytest.raises(ValueError):
        tree_unflatten(td, [1, 2])


def test_tree_map_structure_mismatch() -> None:
    """Test function test_tree_map_structure_mismatch."""
    from ml_switcheroo_compiler.tree_util import tree_map

    with pytest.raises(ValueError):
        tree_map(lambda x, y: x + y, [1, 2], [1, 2, 3])


def test_tree_transpose_mismatch() -> None:
    """Test function test_tree_transpose_mismatch."""
    from ml_switcheroo_compiler.tree_util import tree_structure, tree_transpose

    outer = tree_structure([1, 2])
    inner = tree_structure([1, 2])
    with pytest.raises(ValueError):
        tree_transpose(outer, inner, [1, 2, 3])


def test_tree_all() -> None:
    """Test function test_tree_all."""
    from ml_switcheroo_compiler.tree_util import tree_all

    assert tree_all([True, True]) is True
    assert tree_all([True, False]) is False
    assert tree_all({"a": True}) is True


def test_tree_reduce() -> None:
    """Test function test_tree_reduce."""
    from ml_switcheroo_compiler.tree_util import tree_reduce

    assert as_callable(tree_reduce)(lambda x, y: x + y, [1, 2, 3]) == 6
    assert as_callable(tree_reduce)(lambda x, y: x + y, [1, 2, 3], 10) == 16


def test_tree_transpose_inner_size_zero() -> None:
    """Test function test_tree_transpose_inner_size_zero."""
    from ml_switcheroo_compiler.tree_util import tree_structure, tree_transpose

    outer = tree_structure([1, 2])
    inner = tree_structure([])
    res = tree_transpose(outer, inner, [])
    assert res == []


def test_tree_transpose_inner_loop() -> None:
    """Test function test_tree_transpose_inner_loop."""
    from ml_switcheroo_compiler.tree_util import tree_structure, tree_transpose

    outer = tree_structure([1, 2])
    inner = tree_structure([1, 2])
    res = tree_transpose(outer, inner, [[1, 2], [3, 4]])
    assert res == [[1, 3], [2, 4]]


def test_tree_def_hash_eq() -> None:
    """Test the tree def hash eq behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    (flat1, tree_def1) = as_callable(tree_flatten)({"a": 1, "b": [2, 3]})
    (flat2, tree_def2) = as_callable(tree_flatten)({"a": 1, "b": [2, 3]})
    (flat3, tree_def3) = as_callable(tree_flatten)({"a": 1, "b": [2, 4]})
    (flat4, tree_def4) = as_callable(tree_flatten)({"a": 1, "b": {"c": 3}})
    assert hash(tree_def1) == hash(tree_def2)
    assert tree_def1 == tree_def2
    assert tree_def1 == tree_def3
    assert tree_def1 != tree_def4
    assert tree_def1 != "not a treedef"
