"""Unit tests for MLX distributed collectives including ring reduce-scatter fallback."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


class MockMLXArray:
    """Mock MLX array mimicking mlx.core.array behavior."""

    def __init__(self, data: np.ndarray) -> None:
        """Initialize mock MLX array.

        Args:
            data (np.ndarray): Underlying numerical buffer.
        """
        self._data = np.asarray(data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self._data.ndim

    def __getitem__(self, item: object) -> "MockMLXArray":
        """Slice mock array."""
        return MockMLXArray(self._data[item])  # type: ignore[index]

    def __add__(self, other: "MockMLXArray") -> "MockMLXArray":
        """Elementwise addition."""
        return MockMLXArray(self._data + other._data)

    def __mul__(self, other: "MockMLXArray") -> "MockMLXArray":
        """Elementwise multiplication."""
        return MockMLXArray(self._data * other._data)

    def to_numpy(self) -> np.ndarray:
        """Extract NumPy view."""
        return self._data


class MockMLXModule:
    """Mock MLX core module."""

    @staticmethod
    def add(a: MockMLXArray, b: MockMLXArray) -> MockMLXArray:
        """Add two mock arrays."""
        return a + b

    @staticmethod
    def multiply(a: MockMLXArray, b: MockMLXArray) -> MockMLXArray:
        """Multiply two mock arrays."""
        return a * b

    @staticmethod
    def maximum(a: MockMLXArray, b: MockMLXArray) -> MockMLXArray:
        """Elementwise maximum."""
        return MockMLXArray(np.maximum(a._data, b._data))

    @staticmethod
    def minimum(a: MockMLXArray, b: MockMLXArray) -> MockMLXArray:
        """Elementwise minimum."""
        return MockMLXArray(np.minimum(a._data, b._data))


def test_mlx_ring_reduce_scatter_sum() -> None:
    """Verify MLX ring reduce-scatter host fallback with SUM reduction."""
    # 3 ranks, each holding a (6, 4) tensor
    # Along scatter_dim=0, each chunk is size 2
    r0 = MockMLXArray(np.ones((6, 4), dtype=np.float32) * 1.0)
    r1 = MockMLXArray(np.ones((6, 4), dtype=np.float32) * 2.0)
    r2 = MockMLXArray(np.ones((6, 4), dtype=np.float32) * 3.0)
    all_ranks = [r0, r1, r2]

    # Rank 0 should get chunk 0 reduced: 1 + 2 + 3 = 6
    res0 = _mlx_reduce_scatter(
        MockMLXModule(),
        r0,
        all_ranks_data=all_ranks,
        scatter_dim=0,
        op_type="SUM",
        rank=0,
    )
    assert isinstance(res0, MockMLXArray)
    assert res0.shape == (2, 4)
    np.testing.assert_allclose(res0.to_numpy(), 6.0)

    # Rank 1 should get chunk 1 reduced
    res1 = _mlx_reduce_scatter(
        MockMLXModule(),
        r1,
        all_ranks_data=all_ranks,
        scatter_dim=0,
        op_type="SUM",
        rank=1,
    )
    assert res1.shape == (2, 4)
    np.testing.assert_allclose(res1.to_numpy(), 6.0)


def test_mlx_ring_reduce_scatter_ops() -> None:
    """Verify MLX ring reduce-scatter with PROD, MAX, and MIN operators."""
    r0 = MockMLXArray(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
    r1 = MockMLXArray(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
    all_ranks = [r0, r1]

    # PROD on rank 0 (gets row 0: [2*10, 3*20] = [20, 60])
    res_prod = _mlx_reduce_scatter(
        MockMLXModule(),
        r0,
        all_ranks_data=all_ranks,
        scatter_dim=0,
        op_type="PROD",
        rank=0,
    )
    np.testing.assert_allclose(res_prod.to_numpy(), [[20.0, 60.0]])

    # MAX on rank 1 (gets row 1: [max(4, 30), max(5, 40)] = [30, 40])
    res_max = _mlx_reduce_scatter(
        MockMLXModule(),
        r1,
        all_ranks_data=all_ranks,
        scatter_dim=0,
        op_type="MAX",
        rank=1,
    )
    np.testing.assert_allclose(res_max.to_numpy(), [[30.0, 40.0]])

    # MIN on rank 1 (gets row 1: [min(4, 30), min(5, 40)] = [4, 5])
    res_min = _mlx_reduce_scatter(
        MockMLXModule(),
        r1,
        all_ranks_data=all_ranks,
        scatter_dim=0,
        op_type="MIN",
        rank=1,
    )
    np.testing.assert_allclose(res_min.to_numpy(), [[4.0, 5.0]])

    # Fallback op (else branch) defaults to SUM
    res_fallback = _mlx_reduce_scatter(
        MockMLXModule(),
        r0,
        all_ranks_data=all_ranks,
        scatter_dim=0,
        op_type="UNKNOWN_OP",
        rank=0,
    )
    np.testing.assert_allclose(res_fallback.to_numpy(), [[12.0, 23.0]])


def test_mlx_reduce_scatter_single_tensor_world_size() -> None:
    """Verify single-tensor reduce-scatter slicing when world_size is specified."""
    tensor = MockMLXArray(np.arange(12, dtype=np.float32).reshape(6, 2))

    # World size 3, rank 1 gets rows 2:4
    res = _mlx_reduce_scatter(
        MockMLXModule(),
        tensor,
        world_size=3,
        rank=1,
        scatter_dim=0,
    )
    assert res.shape == (2, 2)
    np.testing.assert_allclose(res.to_numpy(), [[4.0, 5.0], [6.0, 7.0]])

    # World size 1 returns tensor unchanged
    res_1 = _mlx_reduce_scatter(
        MockMLXModule(),
        tensor,
        world_size=1,
        rank=0,
    )
    res_obj: object = res_1
    assert res_obj is tensor


def test_mlx_reduce_scatter_unsupported_exception() -> None:
    """Verify BackendNotSupportedError is raised when no shape or fallback data is given."""
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        _mlx_reduce_scatter(None, "invalid_non_array_tensor")
