"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.common.mixins.array import ArrayASTVisitor
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix.

        Returns:
            EagerValue: Computed output value from `get_fallback_prefix`.
        """
        return "bk"


class DummyVisitor(ArrayASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        ensure_mutable(self)._generator = ensure_eager_tensor(DummyGenerator())


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs or {}


class DummyExpr:
    """Test class DummyExpr."""

    def __init__(self, e: EagerValue) -> None:
        """Test function __init__.

        Args:
            e: Test parameter `e`.
        """
        self.expr = e


def _test_array_mixin_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    vis = DummyVisitor()
    node = DummyNode()

    assert ensure_callable(vis.visit_ApproxMaxK)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_max_k(a, k=2, reduction_dimension=1)[0]"
    assert ensure_callable(vis.visit_ApproxMaxKIndices)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_max_k(a, k=2, reduction_dimension=1)[1]"
    assert ensure_callable(vis.visit_ApproxMinK)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_min_k(a, k=2, reduction_dimension=1)[0]"
    assert ensure_callable(vis.visit_ApproxMinKIndices)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_min_k(a, k=2, reduction_dimension=1)[1]"
    assert ensure_callable(vis.visit_ArgSort)(node, ["a"]) == "bk_argsort(a, dimension=-1)"
    assert ensure_callable(vis.visit_ArgSort)(node, ["a"], dimension=0) == "bk_argsort(a, dimension=0)"
    assert ensure_callable(vis.visit_Argwhere)(node, ["a"]) == "bk_argwhere(a)"
    assert ensure_callable(vis.visit_Argpartition)(node, ["a"], kth=2) == "bk_argpartition(a, kth=2, axis=-1)"
    assert ensure_callable(vis.visit_AsString)(node, ["a"]) == "bk_as_string(a)"
    assert ensure_callable(vis.visit_AxisIndex)(node, ["a"], axis_name="x") == "bk_axis_index(axis_name='x')"

    node_topk = DummyNode({"k": 2})
    assert ensure_callable(vis.visit_TopK)(node_topk, ["a"]) == "bk.sort(a, axis=-1)[..., -(2):][..., ::-1]"
    node_topk_expr = ensure_callable(DummyNode)({"k": ensure_callable(DummyExpr)("k_val"), "return_indices": True})
    assert ensure_callable(vis.visit_TopK)(node_topk_expr, ["a"]) == "bk.argsort(a, axis=-1)[..., -(k_val):][..., ::-1]"

    # Test topk native dispatch
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "jax")
    assert ensure_callable(vis.visit_TopK)(node_topk, ["a"]) == "jax.lax.top_k(a, 2)[0]"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "torch")
    assert ensure_callable(vis.visit_TopK)(node_topk, ["a"]) == "torch.topk(a, 2, dim=-1).values"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "tf")
    assert ensure_callable(vis.visit_TopK)(node_topk_expr, ["a"]) == "tf.math.top_k(a, k=k_val)[1]"
    _state.update(ensure_tensor_dict({"node": node, "node_topk": node_topk, "node_topk_expr": node_topk_expr, "vis": vis}))


def _test_array_mixin_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    node_topk_expr, vis = _state["node_topk_expr"], _state["vis"]
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "keras")
    assert vis.visit_TopK(node_topk_expr, ["a"]) == "keras.ops.top_k(a, k_val)[1]"

    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "bk")
    node_mesh = DummyNode(ensure_eager_tensor({"output_index": 1, "indexing": "xy"}))
    assert ensure_callable(vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "bk.meshgrid(a, b, indexing='xy')[1]"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "mlx")
    assert ensure_callable(vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "mx.meshgrid(a, b, indexing='xy')[1]"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "jax")
    assert ensure_callable(vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "jnp.meshgrid(a, b, indexing='xy')[1]"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "torch")
    assert ensure_callable(vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "torch.meshgrid(a, b, indexing='xy')[1]"

    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "bk")

    node_slice_pos = DummyNode({"dim": 1, "start": 0, "end": 2, "step": 1})
    assert ensure_callable(vis.visit_Slice)(node_slice_pos, ["a"]) == "a[(slice(None),) * (1) + (slice(0, 2, 1),) + (...,)]"
    node_slice_neg = DummyNode({"dim": -2, "start": None, "end": None, "step": None})
    assert ensure_callable(vis.visit_Slice)(node_slice_neg, ["a"]) == "a[(..., slice(None, None, None)) + (slice(None),) * (1)]"

    node_ds = DummyNode({"slice_sizes": [2, 2]})
    assert ensure_callable(vis.visit_DynamicSlice)(node_ds, ["a", "s1", "s2"]) == "a[tuple(slice(s, s + sz) for s, sz in zip([s1, s2], [2, 2]))]"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "jax")
    assert ensure_callable(vis.visit_DynamicSlice)(node_ds, ["a", "s1", "s2"]) == "jax.lax.dynamic_slice(a, (s1, s2,), (2, 2,))"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "tf")
    assert ensure_callable(vis.visit_DynamicSlice)(node_ds, ["a", "s1", "s2"]) == "tf.slice(a, [s1, s2], [2, 2])"
    _state.update(ensure_tensor_dict({"node_ds": node_ds, "node_mesh": node_mesh, "node_slice_neg": node_slice_neg, "node_slice_pos": node_slice_pos}))
    _state.update(ensure_tensor_dict({"node_topk_expr": node_topk_expr, "vis": vis}))


def _test_array_mixin_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    node_ds, vis = _state["node_ds"], _state["vis"]
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "bk")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1", "s2"]) == "(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([s1, s2], upd.shape)), upd), out][1])(a.copy())"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "torch")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1", "s2"]) == "(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([s1, s2], upd.shape)), upd), out][1])(a.clone())"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "jax")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1"]) == "jax.lax.dynamic_update_slice(a, upd, (s1,))"
    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "tf")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1"]) == "tf.tensor_scatter_nd_update(a, tf.stack([s1], axis=-1), upd)"

    ensure_mutable(vis._generator).get_fallback_prefix = ensure_eager_tensor(lambda: "bk")
    node_getitem = DummyNode({"key": "1:3"})
    assert ensure_callable(vis.visit_GetItem)(node_getitem, ["a"]) == "a[1:3]"

    node_paa = DummyNode({"axis": 1})
    assert ensure_callable(vis.visit_PutAlongAxis)(node_paa, ["a", "ind", "val"]) == "bk.put_along_axis(a, ind, val, axis=1)"
    _state.update(ensure_tensor_dict({"node_getitem": node_getitem, "node_paa": node_paa}))


def test_array_mixin() -> None:
    """Test function test_array_mixin."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_array_mixin_part_1(_state)
    _test_array_mixin_part_2(_state)
    _test_array_mixin_part_3(_state)
    assert len(_state) > 0
