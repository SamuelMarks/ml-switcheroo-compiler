"""Tests for PyTree structural utilities and operation schema base dispatch."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph

import ml_switcheroo_compiler.tree_util as tree_util_mod
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, emit_ir_node, get_op
from ml_switcheroo_compiler.tracing.tracer import TracerTape
from tests.testing_utils import coerce_numeric


def test_tree_flatten_unflatten_map_and_reduce() -> None:
    """Verify `tree_flatten`, `tree_unflatten`, `tree_map`, `tree_all`, and `tree_reduce`."""
    from ml_switcheroo_compiler.core.shape import OpValue

    tree: dict[str, OpValue] = {"a": [1, 2, (3, 4)], "b": 5}
    leaves, structure = tree_util_mod.tree_flatten(tree)
    assert leaves == [1, 2, 3, 4, 5]

    reconstructed = tree_util_mod.tree_unflatten(structure, leaves)
    assert reconstructed == {"a": [1, 2, (3, 4)], "b": 5}

    mapped = tree_util_mod.tree_map(lambda x: coerce_numeric(x) * 2, tree)
    assert mapped == {"a": [2, 4, (6, 8)], "b": 10}

    assert tree_util_mod.tree_leaves({"a": 1, "b": 2}) == [1, 2]
    struct_ab = tree_util_mod.tree_structure({"a": 1, "b": 2})
    assert struct_ab.node_type is dict
    assert struct_ab.keys == ["a", "b"]

    assert tree_util_mod.tree_all([True, True]) is True
    assert tree_util_mod.tree_all([True, False]) is False
    assert (
        tree_util_mod.tree_reduce(
            lambda x, y: coerce_numeric(x) + coerce_numeric(y),
            [1, 2, 3],
        )
        == 6
    )
    assert (
        tree_util_mod.tree_reduce(
            lambda x, y: coerce_numeric(x) + coerce_numeric(y),
            [1, 2, 3],
            10,
        )
        == 16
    )


def test_treedef_equality_hashing_and_validation_errors() -> None:
    """Verify `TreeDef` equality, hashing, repr, and `tree_unflatten`/`tree_transpose` error paths."""
    td1 = tree_util_mod.TreeDef(list, [tree_util_mod.TreeDef(type(None))])
    td2 = tree_util_mod.TreeDef(list, [tree_util_mod.TreeDef(type(None))])
    assert td1 == td2
    assert td1 != "not_a_treedef"
    assert hash(td1) == hash(td2)
    assert repr(td1) == "TreeDef(list, [TreeDef(NoneType, [])])"

    with pytest.raises(ValueError, match="All trees must have the same structure"):
        tree_util_mod.tree_map(
            lambda x, y: x + y,
            [1, 2],
            [1, 2, 3],
        )

    with pytest.raises(ValueError, match="Too few leaves"):
        tree_util_mod.tree_unflatten(td1, [])

    with pytest.raises(ValueError, match="Too many leaves"):
        tree_util_mod.tree_unflatten(td1, [1, 2])

    with pytest.raises(ValueError, match="Unsupported treedef node_type"):
        tree_util_mod.tree_unflatten(tree_util_mod.TreeDef(int), [])

    with pytest.raises(ValueError, match="Dict treedef must have keys"):
        tree_util_mod.tree_unflatten(tree_util_mod.TreeDef(dict), [])

    outer_err = tree_util_mod.tree_structure([[1, 2], [3, 4]])
    inner_err = tree_util_mod.tree_structure([1, 2])
    with pytest.raises(ValueError, match="Tree size mismatch"):
        tree_util_mod.tree_transpose(outer_err, inner_err, [1, 2])


def test_tree_mixed_keys_namedtuple_and_transpose() -> None:
    """Verify mixed-type dict keys, NamedTuple reconstruction, multi-tree map, and `tree_transpose`."""
    from ml_switcheroo_compiler.core.shape import OpValue

    mixed_dict: dict[str | int, OpValue] = {1: 10, "a": 20}
    m_leaves, m_def = tree_util_mod.tree_flatten(mixed_dict)
    assert m_leaves == [10, 20]
    assert tree_util_mod.tree_unflatten(m_def, m_leaves) == {1: 10, "a": 20}

    Pair = namedtuple("Pair", ["left", "right"])
    nt_in = Pair(3, 7)
    nt_leaves, nt_def = tree_util_mod.tree_flatten(nt_in)
    assert nt_leaves == [3, 7]
    nt_out = tree_util_mod.tree_unflatten(nt_def, [30, 70])
    assert nt_out == Pair(30, 70)
    assert type(nt_out).__name__ == "Pair"

    summed = tree_util_mod.tree_map(
        lambda a, b: a + b,
        [1, 2],
        [10, 20],
    )
    assert summed == [11, 22]

    outer_td = tree_util_mod.tree_structure([0, 0])
    inner_td = tree_util_mod.tree_structure({"x": 0, "y": 0})
    transposed = tree_util_mod.tree_transpose(
        outer_td,
        inner_td,
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    assert transposed == {"x": [1, 3], "y": [2, 4]}


def test_opdef_schema_emit_ir_node_and_dispatch_eager() -> None:
    """Verify `OpDef` default shape inference, real eager eval, `emit_ir_node` tracing, and `dispatch_eager`."""
    base_op = OpDef()
    assert base_op.dispatch_infer_shape() == ()

    add_op = get_op("Add")()
    with ConfigContext(backend="numpy", eager_mode=True):
        eager_sum = add_op.eager_eval(
            np.array([2.0, 4.0], dtype=np.float32),
            np.array([3.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_allclose(np.asarray(eager_sum), np.array([5.0, 9.0], dtype=np.float32))

    explicit_graph = LogicalGraph(name="explicit_emit")
    nid_explicit = emit_ir_node(explicit_graph, "Add", ["input_0", "input_1"], shape_metadata=(2,))
    assert nid_explicit in explicit_graph.nodes
    assert explicit_graph.nodes[nid_explicit].op_type == "Add"

    tape = TracerTape()
    traced_graph = tape.start_tracing("tape_emit")
    try:
        nid_traced = emit_ir_node(None, "Multiply", ["lhs_0", "rhs_0"], shape_metadata=(2,))
    finally:
        finalized = tape.stop_tracing()
    assert finalized is traced_graph
    assert nid_traced in traced_graph.nodes
    assert traced_graph.nodes[nid_traced].op_type == "Multiply"

    @dispatch_eager("Add")
    def symbolic_add(x: int) -> int:
        """Return symbolic fallback value when not in eager mode.

        Args:
            x (int): Input integer value.

        Returns:
            int: Unmodified integer value.
        """
        return x + 100

    with ConfigContext(eager_mode=False):
        assert symbolic_add(42) == 142
