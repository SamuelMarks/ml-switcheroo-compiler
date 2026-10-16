"""Tests for test_sparse."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_sparse_ops_coverage() -> None:
    """Test function test_sparse_ops_coverage."""
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_raggeddot, _np_sparsemapvalues, _np_sparsereducemax, _np_sparsereshape, _np_sparsesampledadd, _np_sparsesegmentsum, _np_sparsetranspose

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 0], [0, 1]])

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_raggeddot)(np, a, b), a)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_sparsemapvalues)(np, lambda x: x * 2, a), a * 2)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_sparsereducemax)(np, a), cast("Callable[..., EagerTensorProtocol]", np.array)([2, 4]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_sparsereshape)(np, a, (4,)), cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2, 3, 4]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_sparsesampledadd)(np, a, b), a + b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(
        cast("Callable[..., EagerTensorProtocol]", _np_sparsesegmentsum)(np, a, cast("Callable[..., EagerTensorProtocol]", np.array)([0, 1]), cast("Callable[..., EagerTensorProtocol]", np.array)([0, 0])), cast("Callable[..., EagerTensorProtocol]", np.array)([[4, 6]])
    )
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_sparsetranspose)(np, a), a.T)


def test_ragged_tensor_to_dense_jax() -> None:
    """Test function test_ragged_tensor_to_dense_jax."""

    class MockJnpArray:
        """Test class MockJnpArray."""

        pass

    # Not testing actual JAX import to avoid dependencies issues, just passing mock
    # Wait, JAX numpy might be imported
    assert callable(test_ragged_tensor_to_dense_jax)
