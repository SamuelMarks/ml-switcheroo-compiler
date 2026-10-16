"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.sparse_tensor import SparseTensor, SparseTensorCOO, SparseTensorCSR

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.dtype = "float32"
        self.device = "cpu"
        self.requires_grad = True


def test_sparse_tensor() -> None:
    """Test function test_sparse_tensor."""
    t = DummyTensor()

    st = cast("Callable[..., EagerTensorProtocol]", SparseTensor)(t, (10, 10))
    assert st.shape == (10, 10)
    assert st.dtype == "float32"
    assert st.device == "cpu"
    assert bool(st.requires_grad) is True
    assert st.format == "base"

    coo = cast("Callable[..., EagerTensorProtocol]", SparseTensorCOO)(t, t, (10, 10))
    assert coo.indices == t
    assert coo.format == "coo"

    csr = cast("Callable[..., EagerTensorProtocol]", SparseTensorCSR)(t, t, t, (10, 10))
    assert csr.row_pointers == t
    assert csr.column_indices == t
    assert csr.format == "csr"
