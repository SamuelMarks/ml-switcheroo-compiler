# ruff: noqa: E501
"""Test advanced tensors."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
from ml_switcheroo_compiler.core.sparse_tensor import SparseTensorCOO
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.tensor_array import TensorArray


def test_sparse_tensor() -> None:
    """Test the sparse tensor behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Test SparseTensor."
    indices = Tensor(np.array([[0, 0], [1, 2]]), TensorConfig((2, 2), "int32", None))
    values = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", None))
    sp = SparseTensorCOO(indices, values, (3, 3))
    assert sp.dense_shape == (3, 3)


def test_ragged_tensor() -> None:
    """Test the ragged tensor behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Test RaggedTensor."
    values = Tensor(np.array([1.0, 2.0, 3.0]), TensorConfig((3,), "float32", None))
    row_splits = Tensor(np.array([0, 2, 3]), TensorConfig((3,), "int32", None))
    rt = RaggedTensor(values, row_splits)
    assert rt.values is not None


def test_tensor_array_eager() -> None:
    """Test the tensor array eager behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Test TensorArray eager."
    with ConfigContext(eager_mode=True):
        ta = TensorArray(size=2, element_shape=(2, 2), dtype="float32")
        index = Tensor(np.array(0), TensorConfig((), "int32", None))
        val = Tensor(np.ones((2, 2)), TensorConfig((2, 2), "float32", None))
        ta.write(index, val)
        res = ta.read(index)

        index1 = Tensor(np.array(1), TensorConfig((), "int32", None))
        val1 = Tensor(np.zeros((2, 2)), TensorConfig((2, 2), "float32", None))
        ta.write(index1, val1)
        stacked = ta.stack()

        assert res.shape == (2, 2)
        assert stacked.shape == (2, 2, 2)
        assert np.array_equal(res.data, np.ones((2, 2)))
