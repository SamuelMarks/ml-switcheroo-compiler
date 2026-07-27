"""Test module."""

from ml_switcheroo_compiler.core.sparse_tensor import SparseTensor, SparseTensorCOO, SparseTensorCSR


class DummyTensor:
    def __init__(self):
        self.dtype = "float32"
        self.device = "cpu"
        self.requires_grad = True


def test_sparse_tensor():
    t = DummyTensor()

    st = SparseTensor(t, (10, 10))
    assert st.shape == (10, 10)
    assert st.dtype == "float32"
    assert st.device == "cpu"
    assert st.requires_grad is True
    assert st.format == "base"

    coo = SparseTensorCOO(t, t, (10, 10))
    assert coo.indices == t
    assert coo.format == "coo"

    csr = SparseTensorCSR(t, t, t, (10, 10))
    assert csr.row_pointers == t
    assert csr.column_indices == t
    assert csr.format == "csr"
