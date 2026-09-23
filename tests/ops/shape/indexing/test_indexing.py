# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.indexing import (
    DynamicIndexInDim,
    DynamicPartition,
    DynamicSliceInDim,
    DynamicStitch,
    DynamicUpdateIndexInDim,
    DynamicUpdateSliceInDim,
    Extract,
    ExtractVolumePatches,
    IndexSpec,
    ScatterApply,
    ScatterMax,
    ScatterMin,
    ScatterMul,
    SliceInDim,
    TensorScatterSub,
    UnravelIndex,
    boolean_mask,
    gather,
    gather_nd,
    invert_permutation,
    put_along_axis,
    searchsorted,
    select,
    take,
    take_along_axis,
    where,
)
from ml_switcheroo_compiler.ops.shape.indexing import PutAlongAxis as PutAlongAxisClass


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_gather(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="gather")
    assert gather(t, 0, idx) == "gather"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((1, 3))
    res = gather(t, 0, idx)
    assert res.config.shape == (1, 3)


def test_gather_nd(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="gather_nd")
    assert gather_nd(t, idx) == "gather_nd"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((1,))
    assert gather_nd(t, idx).config.shape == (1,)


def test_take(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="take")
    assert take(t, idx) == "take"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((1,))
    assert take(t, idx).config.shape == (1,)


def test_take_along_axis(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    assert take_along_axis(t, idx, 0) == "res"


def test_searchsorted(mocker):
    a = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    v = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="searchsorted")
    assert searchsorted(a, v) == "searchsorted"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((1,))
    assert searchsorted(a, v).config.shape == (1,)


def test_where(mocker):
    c = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "bool", "cpu"))
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="where")
    assert where(c, t, t) == "where"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((2, 3))
    assert where(c, t, t).config.shape == (2, 3)


def test_select(mocker):
    c = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "bool", "cpu"))
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="where")
    assert select(c, t, t) == "where"


def test_boolean_mask(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    c = Tensor(MockTensor((2,)).data, TensorConfig((2,), "bool", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="mask")
    assert boolean_mask(t, c) == "mask"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((2, 3))
    assert boolean_mask(t, c).config.shape == (2, 3)


def test_invert_permutation(mocker):
    t = Tensor(MockTensor((2,)).data, TensorConfig((2,), "int32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="invert")
    assert invert_permutation(t) == "invert"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = "res"
    mock_backend.array.return_value = MockTensor((2,))
    assert invert_permutation(t).config.shape == (2,)


def test_put_along_axis(mocker):
    arr = Tensor(MockTensor((2,)).data, TensorConfig((2,), "float32", "cpu"))
    idx = Tensor(MockTensor((1,)).data, TensorConfig((1,), "int32", "cpu"))
    val = Tensor(MockTensor((1,)).data, TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", return_value="put")
    assert put_along_axis(arr, idx, val, 0) == "put"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.indexing.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2,))
    assert put_along_axis(arr, idx, val, 0).config.shape == (2,)


def test_classes_infer_shape():
    assert Extract().infer_shape() == (None,)
    assert DynamicPartition().infer_shape(None, None, 2) == ()
    assert DynamicStitch().infer_shape(None, None) == ()
    assert TensorScatterSub().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)
    assert ExtractVolumePatches().infer_shape(None, [1], [1], "VALID") == ()
    assert UnravelIndex().infer_shape(None, None) == ()
    assert DynamicSliceInDim().infer_shape(MockTensor((2, 3)), None, 1, axis=0) == (1, 3)
    assert DynamicSliceInDim().infer_shape(None, None, 1) == ()
    assert DynamicUpdateSliceInDim().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)
    assert DynamicIndexInDim().infer_shape(MockTensor((2, 3)), None, axis=0, keepdims=True) == (1, 3)
    assert DynamicIndexInDim().infer_shape(MockTensor((2, 3)), None, axis=0, keepdims=False) == (3,)
    assert DynamicIndexInDim().infer_shape(None, None) == ()
    assert DynamicUpdateIndexInDim().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)
    spec = IndexSpec(0, 2, 1, axis=0)
    assert SliceInDim().infer_shape(MockTensor((4, 3)), spec) == (2, 3)
    assert SliceInDim().infer_shape(None, spec) == ()
    assert ScatterApply().infer_shape(MockTensor((2, 3)), None, None, None) == (2, 3)
    assert ScatterMax().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)
    assert ScatterMin().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)
    assert ScatterMul().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)
    assert PutAlongAxisClass().infer_shape(MockTensor((2, 3)), None, None) == (2, 3)


def test_indexing_exact_shapes(mocker) -> None:
    """Verify exact output shapes for take_along_axis, gather_nd, take, where, and classes.

    Args:
        mocker (object): Pytest mocker fixture.
    """
    from ml_switcheroo_compiler.ops.shape.indexing import (
        gather,
        gather_nd,
        take,
        where,
    )

    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, object], shape: tuple[int, ...], dtype: object) -> str:
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t_src = Tensor(None, TensorConfig((4, 8, 16), "float32", "cpu"))
    t_idx = Tensor(None, TensorConfig((4, 2, 16), "int32", "cpu"))

    # 1. gather (take_along_axis)
    gather(t_src, axis=1, index=t_idx)
    assert captured_shapes["Gather"] == (4, 2, 16)

    # 2. gather_nd
    t_data = Tensor(None, TensorConfig((10, 20, 30), "float32", "cpu"))
    t_nd_idx = Tensor(None, TensorConfig((5, 2), "int32", "cpu"))
    gather_nd(t_data, t_nd_idx)
    assert captured_shapes["GatherNd"] == (5, 30)

    # 3. take along axis
    t_take_idx = Tensor(None, TensorConfig((3,), "int32", "cpu"))
    take(t_src, t_take_idx, axis=1)
    assert captured_shapes["Take"] == (4, 3, 16)

    # 4. where broadcasting
    c = Tensor(None, TensorConfig((1, 8, 1), "bool", "cpu"))
    a = Tensor(None, TensorConfig((4, 1, 16), "float32", "cpu"))
    b = Tensor(None, TensorConfig((4, 8, 1), "float32", "cpu"))
    where(c, a, b)
    assert captured_shapes["Where"] == (4, 8, 16)

    # 5. ExtractVolumePatches
    vol_in = MockTensor((1, 10, 10, 10, 3))
    vol_out = ExtractVolumePatches().infer_shape(vol_in, [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], "VALID")
    assert vol_out == (1, 8, 8, 8, 81)

    # 6. DynamicPartition with valid tensor shapes
    part_data = MockTensor((10, 4))
    part_ids = MockTensor((10,))
    p_shapes = DynamicPartition().infer_shape(part_data, part_ids, 3)
    assert len(p_shapes) == 3
    assert p_shapes[0] == (None, 4)

    # 7. UnravelIndex
    flat_idx = MockTensor((15,))
    unravelled = UnravelIndex().infer_shape(flat_idx, [3, 5])
    assert unravelled == ((15,), (15,))


def test_indexing_edge_cases(mocker) -> None:
    """Test edge cases and fallback branches in shape indexing operations.

    Args:
        mocker (object): Pytest mocker fixture.
    """
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, object], shape: tuple[int, ...], dtype: object) -> str:
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.indexing._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    # 1. gather_nd with empty shape (fallback to in_shape)
    t_empty = Tensor(None, TensorConfig((), "float32", "cpu"))
    t_idx_empty = Tensor(None, TensorConfig((), "int32", "cpu"))
    gather_nd(t_empty, t_idx_empty)
    assert captured_shapes["GatherNd"] == ()

    # 2. take with out-of-bounds axis (both positive and negative)
    t_src = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    t_idx = Tensor(None, TensorConfig((1,), "int32", "cpu"))
    take(t_src, t_idx, axis=10)
    assert captured_shapes["Take"] == (2, 3)
    take(t_src, t_idx, axis=-10)
    assert captured_shapes["Take"] == (2, 3)

    # 3. where with broadcasting error fallback
    c_incompat = Tensor(None, TensorConfig((2,), "bool", "cpu"))
    a_incompat = Tensor(None, TensorConfig((3,), "float32", "cpu"))
    where(c_incompat, a_incompat, a_incompat)
    assert captured_shapes["Where"] == (3,)

    # where with broadcasting error fallback when i_shape is empty
    a_empty = Tensor(None, TensorConfig((), "float32", "cpu"))
    where(c_incompat, a_empty, a_empty)
    assert captured_shapes["Where"] == (2,)

    # 4. DynamicStitch infer_shape variations (indices, data)
    stitch = DynamicStitch()
    # len(d_shape) >= len(idx_shape)
    res1 = stitch.infer_shape([MockTensor((2,))], [MockTensor((2, 3, 4))])
    assert res1 == (None, 3, 4)
    # len(d_shape) < len(idx_shape)
    res2 = stitch.infer_shape([MockTensor((2, 3, 4))], [MockTensor((2,))])
    assert res2 == (None,)
    # elements without shape attribute
    assert stitch.infer_shape([1], [1]) == ()

    # 5. ExtractVolumePatches SAME padding and non-5D input
    evp = ExtractVolumePatches()
    vol_in = MockTensor((1, 10, 10, 10, 3))
    res_same = evp.infer_shape(vol_in, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1], "SAME")
    assert res_same == (1, 5, 5, 5, 81)
    res_non_5d = evp.infer_shape(MockTensor((1, 10, 10, 3)), [1, 3, 3, 1], [1, 1, 1, 1], "VALID")
    assert res_non_5d == ()
