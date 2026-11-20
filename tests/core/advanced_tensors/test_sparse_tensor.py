"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.sparse_tensor import SparseTensor, SparseTensorCOO, SparseTensorCSR
from tests.testing_utils import MockTensorStub

if typing.TYPE_CHECKING:
    pass


class DummyTensor(MockTensorStub):
    """Test class DummyTensor."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.dtype = "float32"
        self.device = "cpu"
        self.requires_grad = True


def test_sparse_tensor() -> None:
    """Test function test_sparse_tensor."""
    t = DummyTensor()

    st = SparseTensor(t, (10, 10))
    assert st.shape == (10, 10)
    assert st.dtype == "float32"
    assert st.device == "cpu"
    assert bool(st.requires_grad) is True
    assert st.format == "base"

    coo = SparseTensorCOO(t, t, (10, 10))
    assert coo.indices is t
    assert coo.format == "coo"

    csr = SparseTensorCSR(t, t, t, (10, 10))
    assert csr.row_pointers is t
    assert csr.column_indices is t
    assert csr.format == "csr"
