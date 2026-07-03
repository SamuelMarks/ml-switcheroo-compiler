"""Test advanced tensors."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
from ml_switcheroo_compiler.core.sparse_tensor import SparseTensorCOO
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.tensor_array import TensorArray
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_sparse_tensor() -> None:
    """Test SparseTensor."""
    indices = Tensor(np.array([[0, 0], [1, 2]]), TensorConfig((2, 2), "int32", None))
    values = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", None))
    sp = SparseTensorCOO(indices, values, (3, 3))
    assert sp.dense_shape == (3, 3)


def test_ragged_tensor() -> None:
    """Test RaggedTensor."""
    values = Tensor(np.array([1.0, 2.0, 3.0]), TensorConfig((3,), "float32", None))
    row_splits = Tensor(np.array([0, 2, 3]), TensorConfig((3,), "int32", None))
    rt = RaggedTensor(values, row_splits)
    assert rt.values is not None


def test_tensor_array_eager() -> None:
    """Test TensorArray eager."""
    # We only test the tracing logic since TensorArray operations emit LogicalNodes
    with ConfigContext(eager_mode=False):
        ta = TensorArray(size=2, element_shape=(2, 2), dtype="float32")

        global_tracing_state.start_tracing()

        index = Tensor(ProxyTensor(id="idx", shape=(), dtype="int32"), TensorConfig((), "int32", None))
        val = Tensor(
            ProxyTensor(id="val", shape=(2, 2), dtype="float32"),
            TensorConfig((2, 2), "float32", None),
        )
        ta.write(index, val)
        res = ta.read(index)
        stacked = ta.stack()
        assert res.shape == (2, 2)
        assert stacked.shape == (2, 2, 2)
        global_tracing_state.stop_tracing()
