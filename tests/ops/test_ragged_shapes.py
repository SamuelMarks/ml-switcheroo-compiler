"""Unit tests for ragged operations shape inference with symbolic dimensions."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.ragged import (
    BooleanMask,
    MapFlatValues,
    RaggedAdd,
    RaggedConstant,
    RaggedCrossHashed,
    RaggedDot,
    RaggedDynamicBroadcast,
    RaggedGather,
    RaggedMatMul,
    RaggedRange,
    RaggedRowSplitsToSegmentIds,
    RaggedSegmentIdsToRowSplits,
    RaggedStack,
    RaggedStackDynamicPartitions,
    RaggedTensorToDense,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _make_tensor(shape: tuple[EagerValue, ...]) -> Tensor:
    """Helper to create dummy tensor with specified shape.

    Args:
        shape (tuple[object, ...]): Shape of tensor.

    Returns:
        Tensor: Dummy tensor instance.
    """
    return typing.cast("Tensor", typing.cast("EagerCallableProtocol", Tensor)(None, typing.cast("EagerCallableProtocol", TensorConfig)(shape, "float32", "cpu")))


def test_ragged_dot_and_matmul_shape_inference() -> None:
    """Verify RaggedDot and RaggedMatMul shape inference."""
    op_dot = RaggedDot()
    op_matmul = RaggedMatMul()

    # RaggedDot: (B=4, var, K=8) x (K=8, N=16) -> (4, var, 16)
    var_len = SymVar("seq_len")
    rt_a = _make_tensor((4, var_len, 8))
    weights = _make_tensor((8, 16))
    out_dot = op_dot.infer_shape(rt_a, weights)
    assert out_dot == (4, var_len, 16)

    # RaggedDot fallback when inner dim is unknown
    rt_a_no_inner = _make_tensor((4,))
    out_dot_fallback = op_dot.infer_shape(rt_a_no_inner, weights)
    assert typing.cast("EagerTensorProtocol", getattr(out_dot_fallback, "__getitem__")(0)) == 4
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_dot_fallback, "__getitem__")(1)), SymVar)
    assert typing.cast("EagerTensorProtocol", getattr(out_dot_fallback, "__getitem__")(2)) == 16

    # RaggedMatMul: (B=2, M=var, K=10) x (B=2, K=10, N=5) -> (2, var, 5)
    rt_mat_a = _make_tensor((2, var_len, 10))
    rt_mat_b = _make_tensor((2, 10, 5))
    out_matmul = op_matmul.infer_shape(rt_mat_a, rt_mat_b)
    assert out_matmul == (2, var_len, 5)

    # Fallbacks
    assert op_dot.infer_shape() == ()
    assert op_matmul.infer_shape() == ()


def test_ragged_tensor_to_dense_and_gather() -> None:
    """Verify RaggedTensorToDense and RaggedGather shape inference."""
    op_dense = RaggedTensorToDense()
    op_gather = RaggedGather()

    # Explicit dense target shape
    rt = _make_tensor((4, SymVar("var"), 16))
    assert op_dense.infer_shape(rt, shape=(4, 32, 16)) == (4, 32, 16)

    # Inferred dense shape with SymVar max_seq_len
    out_inferred = op_dense.infer_shape(rt)
    assert typing.cast("EagerTensorProtocol", getattr(out_inferred, "__getitem__")(0)) == 4
    assert typing.cast("EagerTensorProtocol", getattr(out_inferred, "__getitem__")(2)) == 16
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_inferred, "__getitem__")(1)), SymVar)

    # RaggedGather: params (100, 32), indices (8, 4) -> (8, 4, 32)
    params = _make_tensor((100, 32))
    indices = _make_tensor((8, 4))
    assert op_gather.infer_shape(params, indices) == (8, 4, 32)

    # Fallbacks
    assert op_dense.infer_shape() == ()
    assert op_gather.infer_shape() == ()


def test_ragged_add_and_dynamic_broadcast() -> None:
    """Verify RaggedAdd and RaggedDynamicBroadcast shape inference."""
    op_add = RaggedAdd()
    op_bcast = RaggedDynamicBroadcast()

    rt_a = _make_tensor((4, SymVar("items"), 8))
    rt_b = _make_tensor((4, SymVar("items"), 8))

    assert op_add.infer_shape(rt_a, rt_b) == (4, SymVar("items"), 8)
    assert op_bcast.infer_shape(rt_a, shape=(4, 10, 8)) == (4, 10, 8)
    assert op_bcast.infer_shape(rt_a) == (4, SymVar("items"), 8)

    # Fallbacks
    assert op_add.infer_shape() == ()
    assert op_bcast.infer_shape() == ()


def test_ragged_constant_cross_and_range() -> None:
    """Verify RaggedConstant, RaggedCrossHashed, and RaggedRange shape inference."""
    op_const = RaggedConstant()
    op_cross = RaggedCrossHashed()
    op_range = RaggedRange()

    # RaggedConstant with list of 3 rows
    out_c = op_const.infer_shape([[1, 2], [3], [4, 5, 6]])
    assert typing.cast("EagerTensorProtocol", getattr(out_c, "__getitem__")(0)) == 3
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_c, "__getitem__")(1)), SymVar)

    # RaggedCrossHashed with 2 inputs of batch 5
    in_1 = _make_tensor((5, SymVar("len1")))
    in_2 = _make_tensor((5, SymVar("len2")))
    out_cross = op_cross.infer_shape(in_1, in_2)
    assert typing.cast("EagerTensorProtocol", getattr(out_cross, "__getitem__")(0)) == 5
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_cross, "__getitem__")(1)), SymVar)

    # RaggedRange with 4 start values
    starts = _make_tensor((4,))
    out_range = op_range.infer_shape(starts, None, None)
    assert typing.cast("EagerTensorProtocol", getattr(out_range, "__getitem__")(0)) == 4
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_range, "__getitem__")(1)), SymVar)

    # Fallbacks
    assert op_const.infer_shape() == ()
    assert op_cross.infer_shape() == ()


def test_ragged_splits_and_segments_conversion() -> None:
    """Verify RowSplitsToSegmentIds and SegmentIdsToRowSplits shape inference."""
    op_splits2seg = RaggedRowSplitsToSegmentIds()
    op_seg2splits = RaggedSegmentIdsToRowSplits()

    splits = _make_tensor((5,))  # 4 rows
    out_seg = op_splits2seg.infer_shape(splits)
    assert typing.cast("EagerCallableProtocol", len)(out_seg) == 1
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_seg, "__getitem__")(0)), SymVar)

    # SegmentIdsToRowSplits with known num_segments=4
    seg_ids = _make_tensor((20,))
    out_splits = op_seg2splits.infer_shape(seg_ids, num_segments=4)
    assert out_splits == (5,)

    # SegmentIdsToRowSplits with symbolic num_segments
    out_splits_sym = op_seg2splits.infer_shape(seg_ids)
    assert typing.cast("EagerCallableProtocol", len)(out_splits_sym) == 1
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_splits_sym, "__getitem__")(0)), SymVar)


def test_ragged_stack_and_dynamic_partitions() -> None:
    """Verify RaggedStack and RaggedStackDynamicPartitions shape inference."""
    op_stack = RaggedStack()
    op_part = RaggedStackDynamicPartitions()

    # Stack 3 tensors along axis 0
    t1 = _make_tensor((4, 8))
    t2 = _make_tensor((4, 8))
    t3 = _make_tensor((4, 8))
    assert op_stack.infer_shape(t1, t2, t3, axis=0) == (3, 4, 8)
    assert op_stack.infer_shape(t1, t2, t3, axis=1) == (4, 3, 8)

    # Dynamic partitions into 5 partitions
    data = _make_tensor((100, 16))
    out_part = op_part.infer_shape(data, num_partitions=5)
    assert typing.cast("EagerTensorProtocol", getattr(out_part, "__getitem__")(0)) == 5
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_part, "__getitem__")(1)), SymVar)
    assert typing.cast("EagerTensorProtocol", getattr(out_part, "__getitem__")(2)) == 16

    # Fallbacks
    assert op_stack.infer_shape() == ()


def test_boolean_mask_and_map_flat_values() -> None:
    """Verify BooleanMask and MapFlatValues shape inference."""
    op_mask = BooleanMask()
    op_map = MapFlatValues()

    data = _make_tensor((10, 20, 30))
    mask = _make_tensor((10, 20))  # 2D mask reduces first 2 dims
    out_mask = op_mask.infer_shape(data, mask)
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(out_mask, "__getitem__")(0)), SymVar)
    assert typing.cast("EagerTensorProtocol", getattr(out_mask, "__getitem__")(1)) == 30

    # MapFlatValues preserves shape of primary input
    rt = _make_tensor((4, SymVar("seq"), 8))
    assert typing.cast("EagerCallableProtocol", op_map.infer_shape)(None, rt) == (4, typing.cast("EagerCallableProtocol", SymVar)("seq"), 8)

    # Fallbacks
    assert isinstance(typing.cast("EagerTensorProtocol", getattr(op_mask.infer_shape(), "__getitem__")(0)), SymVar)
    assert op_map.infer_shape(None) == ()


def test_ragged_additional_edge_cases() -> None:
    """Test remaining ragged operations branch paths and edge cases."""
    op_dense = RaggedTensorToDense()
    op_matmul = RaggedMatMul()
    op_const = RaggedConstant()
    op_cross = RaggedCrossHashed()
    op_stack = RaggedStack()
    op_mask = BooleanMask()

    # 1. RaggedTensorToDense with empty input shape (line 78)
    assert op_dense.infer_shape(_make_tensor(())) == ()

    # 2. RaggedMatMul with 1D inputs (line 190)
    t_1d = _make_tensor((4,))
    assert op_matmul.infer_shape(t_1d, t_1d) == (4,)

    # 3. RaggedConstant with non-list/tuple value (line 254)
    assert op_const.infer_shape(123) == ()

    # 4. RaggedCrossHashed with explicit empty inputs kwarg (line 281)
    assert op_cross.infer_shape(inputs=[]) == ()

    # 5. RaggedStack with empty values and negative axis (lines 395, 404)
    assert op_stack.infer_shape(values=[]) == ()
    t_a = _make_tensor((4, 8))
    t_b = _make_tensor((4, 8))
    assert op_stack.infer_shape(t_a, t_b, axis=-1) == (4, 8, 2)

    # 6. BooleanMask with mask=None and mask without shape (branches 486->490, 488->490)
    data_3d = _make_tensor((10, 20, 30))
    res_no_mask = op_mask.infer_shape(data_3d, mask=None)
    assert res_no_mask == (SymVar("masked_count"), 20, 30)

    class DummyMaskNoShape:
        """Dummy mask without shape attributes."""

    res_dummy_mask = typing.cast("EagerCallableProtocol", op_mask.infer_shape)(data_3d, typing.cast("EagerCallableProtocol", DummyMaskNoShape)())
    assert res_dummy_mask == (SymVar("masked_count"), 20, 30)
