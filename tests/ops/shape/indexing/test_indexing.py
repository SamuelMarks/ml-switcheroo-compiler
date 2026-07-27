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
