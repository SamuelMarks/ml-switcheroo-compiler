"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.common.mixins.array import ArrayASTVisitor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "bk"


class DummyVisitor(ArrayASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        setattr(self, "_generator", DummyGenerator())


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__."""
        self.attributes = attrs or {}


class DummyExpr:
    """Test class DummyExpr."""

    def __init__(self, e: EagerValue) -> None:
        """Test function __init__."""
        self.expr = e


def _test_array_mixin_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    vis = DummyVisitor()
    node = DummyNode()

    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ApproxMaxK)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_max_k(a, k=2, reduction_dimension=1)[0]"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ApproxMaxKIndices)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_max_k(a, k=2, reduction_dimension=1)[1]"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ApproxMinK)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_min_k(a, k=2, reduction_dimension=1)[0]"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ApproxMinKIndices)(node, ["a"], k=2, reduction_dimension=1) == "bk_approx_min_k(a, k=2, reduction_dimension=1)[1]"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ArgSort)(node, ["a"]) == "bk_argsort(a, dimension=-1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ArgSort)(node, ["a"], dimension=0) == "bk_argsort(a, dimension=0)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Argwhere)(node, ["a"]) == "bk_argwhere(a)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Argpartition)(node, ["a"], kth=2) == "bk_argpartition(a, kth=2, axis=-1)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_AsString)(node, ["a"]) == "bk_as_string(a)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_AxisIndex)(node, ["a"], axis_name="x") == "bk_axis_index(axis_name='x')"

    node_topk = DummyNode({"k": 2})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TopK)(node_topk, ["a"]) == "bk.sort(a, axis=-1)[..., -(2):][..., ::-1]"
    node_topk_expr = cast("Callable[..., EagerTensorProtocol]", DummyNode)({"k": cast("Callable[..., EagerTensorProtocol]", DummyExpr)("k_val"), "return_indices": True})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TopK)(node_topk_expr, ["a"]) == "bk.argsort(a, axis=-1)[..., -(k_val):][..., ::-1]"

    # Test topk native dispatch
    setattr(vis._generator, "get_fallback_prefix", lambda: "jax")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TopK)(node_topk, ["a"]) == "jax.lax.top_k(a, 2)[0]"
    setattr(vis._generator, "get_fallback_prefix", lambda: "torch")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TopK)(node_topk, ["a"]) == "torch.topk(a, 2, dim=-1).values"
    setattr(vis._generator, "get_fallback_prefix", lambda: "tf")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TopK)(node_topk_expr, ["a"]) == "tf.math.top_k(a, k=k_val)[1]"
    getattr(_state, "update")({"node": node, "node_topk": node_topk, "node_topk_expr": node_topk_expr, "vis": vis})
    _state.update(locals())


def _test_array_mixin_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    node_topk_expr, vis = _state["node_topk_expr"], _state["vis"]
    setattr(vis._generator, "get_fallback_prefix", lambda: "keras")
    assert vis.visit_TopK(node_topk_expr, ["a"]) == "keras.ops.top_k(a, k_val)[1]"

    setattr(vis._generator, "get_fallback_prefix", lambda: "bk")
    node_mesh = DummyNode({"output_index": 1, "indexing": "xy"})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "bk.meshgrid(a, b, indexing='xy')[1]"
    setattr(vis._generator, "get_fallback_prefix", lambda: "mlx")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "mx.meshgrid(a, b, indexing='xy')[1]"
    setattr(vis._generator, "get_fallback_prefix", lambda: "jax")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "jnp.meshgrid(a, b, indexing='xy')[1]"
    setattr(vis._generator, "get_fallback_prefix", lambda: "torch")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Meshgrid)(node_mesh, ["a", "b"]) == "torch.meshgrid(a, b, indexing='xy')[1]"

    setattr(vis._generator, "get_fallback_prefix", lambda: "bk")

    node_slice_pos = DummyNode({"dim": 1, "start": 0, "end": 2, "step": 1})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Slice)(node_slice_pos, ["a"]) == "a[(slice(None),) * (1) + (slice(0, 2, 1),) + (...,)]"
    node_slice_neg = DummyNode({"dim": -2, "start": None, "end": None, "step": None})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Slice)(node_slice_neg, ["a"]) == "a[(..., slice(None, None, None)) + (slice(None),) * (1)]"

    node_ds = DummyNode({"slice_sizes": [2, 2]})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_DynamicSlice)(node_ds, ["a", "s1", "s2"]) == "a[tuple(slice(s, s + sz) for s, sz in zip([s1, s2], [2, 2]))]"
    setattr(vis._generator, "get_fallback_prefix", lambda: "jax")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_DynamicSlice)(node_ds, ["a", "s1", "s2"]) == "jax.lax.dynamic_slice(a, (s1, s2,), (2, 2,))"
    setattr(vis._generator, "get_fallback_prefix", lambda: "tf")
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_DynamicSlice)(node_ds, ["a", "s1", "s2"]) == "tf.slice(a, [s1, s2], [2, 2])"
    getattr(_state, "update")({"node_ds": node_ds, "node_mesh": node_mesh, "node_slice_neg": node_slice_neg, "node_slice_pos": node_slice_pos})
    _state.update(locals())


def _test_array_mixin_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    node_ds, vis = _state["node_ds"], _state["vis"]
    setattr(vis._generator, "get_fallback_prefix", lambda: "bk")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1", "s2"]) == "(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([s1, s2], upd.shape)), upd), out][1])(a.copy())"
    setattr(vis._generator, "get_fallback_prefix", lambda: "torch")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1", "s2"]) == "(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([s1, s2], upd.shape)), upd), out][1])(a.clone())"
    setattr(vis._generator, "get_fallback_prefix", lambda: "jax")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1"]) == "jax.lax.dynamic_update_slice(a, upd, (s1,))"
    setattr(vis._generator, "get_fallback_prefix", lambda: "tf")
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1"]) == "tf.tensor_scatter_nd_update(a, tf.stack([s1], axis=-1), upd)"

    setattr(vis._generator, "get_fallback_prefix", lambda: "bk")
    node_getitem = DummyNode({"key": "1:3"})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_GetItem)(node_getitem, ["a"]) == "a[1:3]"

    node_paa = DummyNode({"axis": 1})
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_PutAlongAxis)(node_paa, ["a", "ind", "val"]) == "bk.put_along_axis(a, ind, val, axis=1)"
    getattr(_state, "update")({"node_getitem": node_getitem, "node_paa": node_paa})


def test_array_mixin() -> None:
    """Test function test_array_mixin."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_array_mixin_part_1(_state)
    _test_array_mixin_part_2(_state)
    _test_array_mixin_part_3(_state)
    assert isinstance(_state, dict)
