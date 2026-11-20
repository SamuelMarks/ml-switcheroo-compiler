"""Test module."""

from __future__ import annotations

import typing
from collections.abc import Mapping

import pytest
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.common.mixins.array import ArrayASTVisitor
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockNodeStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> BackendArg:
        """Test function get_fallback_prefix.

        Returns:
            BackendArg: Computed output value from `get_fallback_prefix`.
        """
        return "bk"


class DummyVisitor(ArrayASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        pytest.MonkeyPatch().setattr(self, "_generator", DummyGenerator(), raising=False)


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    def __init__(self, attrs: Mapping[str, AttributeValue] | None = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = dict(attrs) if attrs is not None else {}


class DummyExpr:
    """Test class DummyExpr."""

    shape: tuple[int, ...] = ()

    def __init__(self, e: BackendArg) -> None:
        """Test function __init__.

        Args:
            e: Test parameter `e`.
        """
        self.expr = e

    def tobytes(self) -> bytes:
        """Return empty byte buffer for `ArrayBufferProtocol` conformance.

        Returns:
            bytes: Empty byte string.
        """
        return b""


class _TestArrayMixinState(typing.TypedDict, total=False):
    """Typed shared state for `_test_array_mixin` split helpers."""

    node_ds: DummyNode
    node_topk_expr: DummyNode
    vis: DummyVisitor


def _test_array_mixin_part_1(_state: _TestArrayMixinState) -> None:
    """Execute part 1 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    vis = DummyVisitor()
    node = DummyNode()

    assert (
        vis.visit_ApproxMaxK(node, ["a"], k=2, reduction_dimension=1)
        == "bk_approx_max_k(a, k=2, reduction_dimension=1)[0]"
    )
    assert (
        vis.visit_ApproxMaxKIndices(node, ["a"], k=2, reduction_dimension=1)
        == "bk_approx_max_k(a, k=2, reduction_dimension=1)[1]"
    )
    assert (
        vis.visit_ApproxMinK(node, ["a"], k=2, reduction_dimension=1)
        == "bk_approx_min_k(a, k=2, reduction_dimension=1)[0]"
    )
    assert (
        vis.visit_ApproxMinKIndices(node, ["a"], k=2, reduction_dimension=1)
        == "bk_approx_min_k(a, k=2, reduction_dimension=1)[1]"
    )
    assert vis.visit_ArgSort(node, ["a"]) == "bk_argsort(a, dimension=-1)"
    assert vis.visit_ArgSort(node, ["a"], dimension=0) == "bk_argsort(a, dimension=0)"
    assert vis.visit_Argwhere(node, ["a"]) == "bk_argwhere(a)"
    assert vis.visit_Argpartition(node, ["a"], kth=2) == "bk_argpartition(a, kth=2, axis=-1)"
    assert vis.visit_AsString(node, ["a"]) == "bk_as_string(a)"
    assert vis.visit_AxisIndex(node, ["a"], axis_name="x") == "bk_axis_index(axis_name='x')"

    node_topk = DummyNode({"k": 2})
    assert vis.visit_TopK(node_topk, ["a"]) == "bk.sort(a, axis=-1)[..., -(2):][..., ::-1]"
    node_topk_expr = DummyNode({"k": DummyExpr("k_val"), "return_indices": True})
    assert vis.visit_TopK(node_topk_expr, ["a"]) == "bk.argsort(a, axis=-1)[..., -(k_val):][..., ::-1]"

    # Test topk native dispatch
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "jax", raising=False)
    assert vis.visit_TopK(node_topk, ["a"]) == "jax.lax.top_k(a, 2)[0]"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "torch", raising=False)
    assert vis.visit_TopK(node_topk, ["a"]) == "torch.topk(a, 2, dim=-1).values"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "tf", raising=False)
    assert vis.visit_TopK(node_topk_expr, ["a"]) == "tf.math.top_k(a, k=k_val)[1]"
    _state.update({"node_topk_expr": node_topk_expr, "vis": vis})


def _test_array_mixin_part_2(_state: _TestArrayMixinState) -> None:
    """Execute part 2 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    node_topk_expr, vis = _state["node_topk_expr"], _state["vis"]
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "keras", raising=False)
    assert vis.visit_TopK(node_topk_expr, ["a"]) == "keras.ops.top_k(a, k_val)[1]"

    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "bk", raising=False)
    node_mesh = DummyNode({"output_index": 1, "indexing": "xy"})
    assert vis.visit_Meshgrid(node_mesh, ["a", "b"]) == "bk.meshgrid(a, b, indexing='xy')[1]"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "mlx", raising=False)
    assert vis.visit_Meshgrid(node_mesh, ["a", "b"]) == "mx.meshgrid(a, b, indexing='xy')[1]"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "jax", raising=False)
    assert vis.visit_Meshgrid(node_mesh, ["a", "b"]) == "jnp.meshgrid(a, b, indexing='xy')[1]"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "torch", raising=False)
    assert vis.visit_Meshgrid(node_mesh, ["a", "b"]) == "torch.meshgrid(a, b, indexing='xy')[1]"

    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "bk", raising=False)

    node_slice_pos = DummyNode({"dim": 1, "start": 0, "end": 2, "step": 1})
    assert vis.visit_Slice(node_slice_pos, ["a"]) == "a[(slice(None),) * (1) + (slice(0, 2, 1),) + (...,)]"
    node_slice_neg = DummyNode({"dim": -2, "start": None, "end": None, "step": None})
    assert vis.visit_Slice(node_slice_neg, ["a"]) == "a[(..., slice(None, None, None)) + (slice(None),) * (1)]"

    node_ds = DummyNode({"slice_sizes": [2, 2]})
    assert (
        vis.visit_DynamicSlice(node_ds, ["a", "s1", "s2"])
        == "a[tuple(slice(s, s + sz) for s, sz in zip([s1, s2], [2, 2]))]"
    )
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "jax", raising=False)
    assert vis.visit_DynamicSlice(node_ds, ["a", "s1", "s2"]) == "jax.lax.dynamic_slice(a, (s1, s2,), (2, 2,))"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "tf", raising=False)
    assert vis.visit_DynamicSlice(node_ds, ["a", "s1", "s2"]) == "tf.slice(a, [s1, s2], [2, 2])"
    _state.update({"node_ds": node_ds})
    _state.update({"node_topk_expr": node_topk_expr, "vis": vis})


def _test_array_mixin_part_3(_state: _TestArrayMixinState) -> None:
    """Execute part 3 of test_array_mixin.

    Args:
        _state: Shared test state dictionary.
    """
    node_ds, vis = _state["node_ds"], _state["vis"]
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "bk", raising=False)
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1", "s2"]) == (
        "(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([s1, s2], upd.shape)),"
        " upd), out][1])(a.copy())"
    )
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "torch", raising=False)
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1", "s2"]) == (
        "(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([s1, s2], upd.shape)),"
        " upd), out][1])(a.clone())"
    )
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "jax", raising=False)
    assert vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1"]) == "jax.lax.dynamic_update_slice(a, upd, (s1,))"
    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "tf", raising=False)
    assert (
        vis.visit_DynamicUpdateSlice(node_ds, ["a", "upd", "s1"])
        == "tf.tensor_scatter_nd_update(a, tf.stack([s1], axis=-1), upd)"
    )

    pytest.MonkeyPatch().setattr(vis._generator, "get_fallback_prefix", lambda: "bk", raising=False)
    node_getitem = DummyNode({"key": "1:3"})
    assert vis.visit_GetItem(node_getitem, ["a"]) == "a[1:3]"

    node_paa = DummyNode({"axis": 1})
    assert vis.visit_PutAlongAxis(node_paa, ["a", "ind", "val"]) == "bk.put_along_axis(a, ind, val, axis=1)"


def test_array_mixin() -> None:
    """Test function test_array_mixin."""
    _state: _TestArrayMixinState = {}
    _test_array_mixin_part_1(_state)
    _test_array_mixin_part_2(_state)
    _test_array_mixin_part_3(_state)
    assert len(_state) > 0
