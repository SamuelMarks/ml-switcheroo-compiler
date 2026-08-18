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
