"""Unit and algebraic law tests for open PyTreeNodeTypeclass instances and Foldable/Traversable combinators."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pytest
from ml_switcheroo_ir.typeclasses import INT_SUM_MONOID, Err, Invalid, Ok, Result, Semigroup, Valid

import ml_switcheroo_compiler.tree_util as tu
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.sparse_tensor import SparseTensorCOO, SparseTensorCSR
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

_EqT = typing.TypeVar("_EqT")


class _CustomBox:
    """Custom shaped container implementing ShapeProtocol to test open register_pytree_node."""

    def __init__(self, label: str, left: int, right: int) -> None:
        """Initialize _CustomBox.

        Args:
            label (str): Static box tag stored in aux_data.
            left (int): Left integer payload.
            right (int): Right integer payload.
        """
        self.label = label
        self.left = left
        self.right = right
        self.shape: tuple[int, ...] = (2,)

    def __eq__(self, other: _EqT) -> bool:
        """Compare two _CustomBox instances for value equality.

        Args:
            other (_EqT): Candidate value.

        Returns:
            bool: True when label, left, and right match.
        """
        if not isinstance(other, _CustomBox):
            return False
        return self.label == other.label and self.left == other.left and self.right == other.right


def _flatten_box(box: _CustomBox) -> tuple[Sequence[OpValue], tu.AuxData]:
    """Flatten _CustomBox into (left, right) leaves and static label aux_data.

    Args:
        box (_CustomBox): Custom box instance.

    Returns:
        tuple[Sequence[OpValue], tu.AuxData]: Pair of children and label string.
    """
    return (box.left, box.right), box.label


def _unflatten_box(aux: tu.AuxData, children: Sequence[OpValue]) -> _CustomBox:
    """Reconstruct _CustomBox from label aux_data and two integer children.

    Args:
        aux (tu.AuxData): Static string label.
        children (Sequence[OpValue]): Reconstructed child values.

    Returns:
        _CustomBox: Reconstructed _CustomBox.
    """
    left_v = int(children[0]) if isinstance(children[0], int) else 0
    right_v = int(children[1]) if isinstance(children[1], int) else 0
    return _CustomBox(str(aux), left_v, right_v)


def test_treedef_hash_key_sensitivity_and_witnesses() -> None:
    """Verify TreeDef key/aux hash disambiguation and TREE_DEF_{EQ,HASH,PRETTY} witnesses."""
    _, td_a = tu.tree_flatten({"a": 1})
    _, td_b = tu.tree_flatten({"b": 1})
    assert not tu.TREE_DEF_EQ.eq(td_a, td_b)
    assert tu.TREE_DEF_HASH.hash_value(td_a) != tu.TREE_DEF_HASH.hash_value(td_b)
    assert "TreeDef(dict," in tu.TREE_DEF_PRETTY.render(td_a)

    td_aux = tu.TreeDef(_CustomBox, [tu.TreeDef(type(None))], None, "tag_1")
    assert "aux='tag_1'" in repr(td_aux)


def test_builtin_and_custom_pytree_node_instances() -> None:
    """Verify SparseTensorCOO, SparseTensorCSR, RaggedTensor, and custom PyTreeNodeTypeclass roundtrips."""
    assert tu.is_registered_pytree_node(SparseTensorCOO)
    assert tu.is_registered_pytree_node(SparseTensorCSR)
    assert tu.is_registered_pytree_node(RaggedTensor)
    assert not tu.is_registered_pytree_node(int)

    t_idx = Tensor([[0, 1]], TensorConfig((1, 2), DType.Int32, None))
    t_val = Tensor([5.0], TensorConfig((1,), DType.Float32, None))
    coo = SparseTensorCOO(t_idx, t_val, (3, 3))
    coo_leaves, coo_def = tu.tree_flatten(coo)
    assert len(coo_leaves) == 2
    coo_rt = tu.tree_unflatten(coo_def, coo_leaves)
    assert isinstance(coo_rt, SparseTensorCOO) and coo_rt.dense_shape == (3, 3)

    coo_empty_aux = tu.TreeDef(SparseTensorCOO, coo_def.children_defs, None, None)
    coo_fallback = tu.tree_unflatten(coo_empty_aux, coo_leaves)
    assert isinstance(coo_fallback, SparseTensorCOO) and coo_fallback.dense_shape == ()

    t_rp = Tensor([0, 1], TensorConfig((2,), DType.Int32, None))
    t_ci = Tensor([1], TensorConfig((1,), DType.Int32, None))
    csr = SparseTensorCSR(t_rp, t_ci, t_val, (1, 3))
    csr_leaves, csr_def = tu.tree_flatten(csr)
    assert len(csr_leaves) == 3
    csr_rt = tu.tree_unflatten(csr_def, csr_leaves)
    assert isinstance(csr_rt, SparseTensorCSR) and csr_rt.dense_shape == (1, 3)

    csr_empty_aux = tu.TreeDef(SparseTensorCSR, csr_def.children_defs, None, None)
    csr_fallback = tu.tree_unflatten(csr_empty_aux, csr_leaves)
    assert isinstance(csr_fallback, SparseTensorCSR) and csr_fallback.dense_shape == ()

    rag = RaggedTensor(t_val, t_rp)
    rag_leaves, rag_def = tu.tree_flatten(rag)
    assert len(rag_leaves) == 2
    rag_rt = tu.tree_unflatten(rag_def, rag_leaves)
    assert isinstance(rag_rt, RaggedTensor)

    witness = tu.register_pytree_node(_CustomBox, _flatten_box, _unflatten_box)
    assert witness.node_type is _CustomBox
    assert tu.is_registered_pytree_node(_CustomBox)

    nested: dict[str, OpValue] = {"box": _CustomBox("alpha", 3, 4), "extra": 10}
    mapped = tu.tree_map(lambda x: (x * 2) if isinstance(x, int) else x, nested)
    assert mapped == {"box": _CustomBox("alpha", 6, 8), "extra": 20}

    erased_entry = tu._PYTREE_REGISTRY[_CustomBox]
    with pytest.raises(TypeError, match="Expected instance of _CustomBox"):
        erased_entry.flatten_erased(999)


def test_functor_foldable_and_traversable_combinators() -> None:
    """Verify Functor laws, tree_fold_map, tree_all_by, tree_any_by, tree_traverse_result, and Validation."""
    tree: dict[str, OpValue] = {"x": [1, 2], "y": (3, 4)}

    # Functor identity and composition laws
    assert tu.tree_map(lambda v: v, tree) == tree

    def _inc(v: OpValue) -> OpValue:
        """Increment integer leaf by 1.

        Args:
            v (OpValue): Leaf value.

        Returns:
            OpValue: Incremented leaf.
        """
        return (v + 1) if isinstance(v, int) else v

    def _mul3(v: OpValue) -> OpValue:
        """Multiply integer leaf by 3.

        Args:
            v (OpValue): Leaf value.

        Returns:
            OpValue: Scaled leaf.
        """
        return (v * 3) if isinstance(v, int) else v

    assert tu.tree_map(lambda v: _mul3(_inc(v)), tree) == tu.tree_map(_mul3, tu.tree_map(_inc, tree))

    ok_map = tu.tree_map_result(_inc, tree)
    assert isinstance(ok_map, Ok) and ok_map.value == {"x": [2, 3], "y": (4, 5)}
    err_map = tu.tree_map_result(lambda a, b: a, [1, 2], [1, 2, 3])
    assert isinstance(err_map, Err)

    total = tu.tree_fold_map(INT_SUM_MONOID, lambda v: v if isinstance(v, int) else 0, tree)
    assert total == 10
    assert tu.tree_all_by(lambda v: isinstance(v, int) and v > 0, tree) is True
    assert tu.tree_all_by(lambda v: isinstance(v, int) and v > 2, tree) is False
    t_any_true: dict[str, OpValue] = {"a": [0, False], "b": 1}
    t_any_false: dict[str, OpValue] = {"a": [0, False], "b": 0}
    assert tu.tree_any(t_any_true) is True
    assert tu.tree_any(t_any_false) is False
    assert tu.tree_any_by(lambda v: v == 3, tree) is True
    assert tu.tree_any_by(lambda v: v == 99, tree) is False

    # Traversable over Result (short-circuits on first Err)
    visited: list[OpValue] = []

    def _step_res(v: OpValue) -> Ok[OpValue] | Err[str]:
        """Double positive ints below 3, failing on 2 to verify short-circuiting.

        Args:
            v (OpValue): Leaf value.

        Returns:
            Ok[OpValue] | Err[str]: Doubled int or error message.
        """
        visited.append(v)
        if isinstance(v, int) and v == 2:
            return Err("hit_two")
        return Ok((v * 2) if isinstance(v, int) else v)

    res_err = tu.tree_traverse_result(_step_res, tree)
    assert isinstance(res_err, Err) and res_err.error == "hit_two"
    assert visited == [1, 2]

    def _add10_ok(v: OpValue) -> Result[str, OpValue]:
        """Return Ok(v + 10) for integer leaves.

        Args:
            v (OpValue): Leaf value.

        Returns:
            Result[str, OpValue]: Ok incremented leaf.
        """
        return Ok((v + 10) if isinstance(v, int) else v)

    res_ok: Result[str, OpValue] = tu.tree_traverse_result(_add10_ok, tree)
    assert isinstance(res_ok, Ok) and res_ok.value == {"x": [11, 12], "y": (13, 14)}

    # Traversable over Validation (accumulates ALL Invalid errors without short-circuiting)
    tuple_sg: Semigroup[tuple[str, ...]] = Semigroup(combine=lambda a, b: a + b)

    def _step_val(v: OpValue) -> Valid[OpValue] | Invalid[tuple[str, ...]]:
        """Reject even integers while accumulating all errors.

        Args:
            v (OpValue): Leaf value.

        Returns:
            Valid[OpValue] | Invalid[tuple[str, ...]]: Validated odd leaf or tuple of error tags.
        """
        if isinstance(v, int) and v % 2 == 0:
            return Invalid((f"even:{v}",))
        return Valid(v)

    val_err = tu.tree_traverse_validation(tuple_sg, _step_val, tree)
    assert isinstance(val_err, Invalid) and val_err.errors == ("even:2", "even:4")

    val_ok = tu.tree_traverse_validation(tuple_sg, lambda v: Valid(v), tree)
    assert isinstance(val_ok, Valid) and val_ok.value == tree
