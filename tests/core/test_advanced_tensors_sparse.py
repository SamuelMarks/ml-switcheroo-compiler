"""Test sparse tensors."""

import numpy as np
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.sparse_tensor import SparseTensor, SparseTensorCOO, SparseTensorCSR


def test_sparse_tensor_coo() -> None:
    """Test SparseTensorCOO."""
    indices = Tensor(np.array([[0, 0], [1, 2]]), TensorConfig((2, 2), "int32", None))
    values = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", None))
    sp = SparseTensorCOO(indices, values, (3, 3))
    assert sp.dense_shape == (3, 3)
    assert sp.format == "coo"
    assert sp.indices is indices


def test_sparse_tensor_csr() -> None:
    """Test SparseTensorCSR."""
    row_pointers = Tensor(np.array([0, 1, 2, 2]), TensorConfig((4,), "int32", None))
    column_indices = Tensor(np.array([0, 2]), TensorConfig((2,), "int32", None))
    values = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", None))
    sp = SparseTensorCSR(row_pointers, column_indices, values, (3, 3))
    assert sp.dense_shape == (3, 3)
    assert sp.format == "csr"
    assert sp.row_pointers is row_pointers
    assert sp.column_indices is column_indices


def test_sparse_tensor_base() -> None:
    """Test SparseTensor."""
    values = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", None))
    sp = SparseTensor(values, (3, 3))
    assert sp.dense_shape == (3, 3)
    assert sp.format == "base"
