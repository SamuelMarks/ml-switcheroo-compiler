"""Module test_slicing.py."""

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.slicing import Choose, IndexInDim, Slice, StridedSlice, UpdateSlice, slice, strided_slice


class MockTensor:
    """MockTensor."""

    def __init__(self, shape=()):
        """__init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = np.zeros(shape)


def test_slice(mocker):
    """test_slice."""
    t = Tensor(MockTensor((5, 5)).data, TensorConfig((5, 5), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.slicing._emit_shape_node", return_value="sliced")
    assert slice(t, 0, 1, 3, 1) == "sliced"
    config.eager_mode = True
    res = slice(t, 0, 1, 3, 1)
    assert res.config.shape == (2, 5)


def test_strided_slice(mocker):
    """test_strided_slice."""
    t = Tensor(MockTensor((5, 5)).data, TensorConfig((5, 5), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.slicing._emit_shape_node", return_value="strided")
    assert strided_slice(t, [1, 1], [3, 3], [1, 1]) == "strided"
    config.eager_mode = True
    res = strided_slice(t, [1, 1], [3, 3], [1, 1])
    assert res.config.shape == (2, 2)


def test_slice_class():
    """test_slice_class."""
    assert Slice().infer_shape() == ()


def test_strided_slice_class():
    """test_strided_slice_class."""
    assert StridedSlice().infer_shape() == ()


def test_choose_class():
    """test_choose_class."""
    op = Choose()
    assert op.infer_shape(MockTensor((2, 3)), None) == (2, 3)

    class NoShape:
        """NoShape."""

        pass

    assert op.infer_shape(NoShape(), None) == ()


def test_index_in_dim_update_slice_coverage():
    """test_index_in_dim_update_slice_coverage."""
    assert IndexInDim().infer_shape(MockTensor((2, 3)), MockTensor((2,))) == (2, 3)
    assert IndexInDim().infer_shape(MockTensor((2, 3)), MockTensor((2,)), axis=1, keepdims=False) == (2, 2)
    assert IndexInDim().infer_shape(MockTensor((2, 3)), MockTensor(()), keepdims=False) == (3,)
    assert IndexInDim().infer_shape(MockTensor(()), MockTensor(())) == ()
    assert UpdateSlice().infer_shape(MockTensor((2, 3))) == (2, 3)
    assert UpdateSlice().infer_shape(MockTensor((2, 3)), MockTensor((2, 3))) == (2, 3)
    assert UpdateSlice().infer_shape(MockTensor(())) == ()


def test_slicing_dispatcher():
    """test_slicing_dispatcher."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.shape.slicing import index_in_dim, update_slice

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"

        assert index_in_dim(1, 2) == "mock_result"
        mock_dispatch.assert_called_with("IndexInDim", 1, 2)

        assert update_slice(1, 2) == "mock_result"
        mock_dispatch.assert_called_with("UpdateSlice", 1, 2)


def test_slice_and_strided_slice_exact_shapes(mocker) -> None:
    """Verify exact output shapes for slice and strided_slice in graph mode.

    Args:
        mocker (object): Pytest mocker fixture.
    """
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, object], shape: tuple[int, ...], dtype: object) -> str:
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.slicing._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t = Tensor(None, TensorConfig((10, 20), "float32", "cpu"))

    # 1. Slice along axis 0 with start, end, step
    slice(t, axis=0, start=2, end=8, step=2)
    assert captured_shapes["Slice"] == (3, 20)

    # 1b. Slice with end=None (defaults to dim_len)
    slice(t, axis=0, start=2, end=None)
    assert captured_shapes["Slice"] == (8, 20)

    # 2. Slice with step > 1 and negative start/end
    slice(t, axis=1, start=-15, end=-5, step=3)
    assert captured_shapes["Slice"] == (10, 4)

    # 3. Strided slice across multiple dimensions with step > 1
    strided_slice(t, begin=[1, 2], end=[9, 18], strides=[2, 4])
    assert captured_shapes["StridedSlice"] == (4, 4)

    # 4. Strided slice with negative stride (reversing dimension)
    strided_slice(t, begin=[8, 15], end=[2, 5], strides=[-2, -3])
    assert captured_shapes["StridedSlice"] == (3, 4)

    # 5. Slice with negative step (reversing dimension)
    slice(t, axis=0, start=8, end=2, step=-2)
    assert captured_shapes["Slice"] == (3, 20)

    # 6. Slice with negative end and positive step
    slice(t, axis=0, start=1, end=-2, step=1)
    assert captured_shapes["Slice"] == (7, 20)

    # 7. Slice with out-of-bounds axis (rank > 0 but norm_axis >= rank)
    slice(t, axis=5, start=0, end=5)
    assert captured_shapes["Slice"] == (10, 20)

    # 8. Slice with rank == 0 scalar tensor
    t_0d = Tensor(None, TensorConfig((), "float32", "cpu"))
    slice(t_0d, axis=0)
    assert captured_shapes["Slice"] == ()

    # 9. IndexInDim with negative axis
    assert IndexInDim().infer_shape(MockTensor((2, 3)), MockTensor((1,)), axis=-1) == (2, 1)

    # 10. UpdateSlice with None operand
    assert UpdateSlice().infer_shape(None) == ()
