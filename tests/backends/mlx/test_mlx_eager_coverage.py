"""Test MLX eager edge cases coverage."""

from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.backends.mlx.eager import _mlx_partition, _mlx_zeros, execute_op


def test_mlx_eager_execute_op(monkeypatch):
    """Test execute_op edge cases in mlx/eager.py."""

    # Test fallback path when op is not in mlx_eager_registry
    class DummyMlxBackend:
        @classmethod
        def execute_numpy_fallback(cls, op_type, *args, **kwargs):
            return "mock_numpy_fallback"

    # Need to monkeypatch _execute_numpy_fallback
    monkeypatch.setattr("ml_switcheroo_compiler.backends.mlx.eager._execute_numpy_fallback", lambda cls, op_type, *args, **kwargs: "mock_numpy_fallback")

    # Simulate AttributeError in mlx
    mock_registry = MagicMock()
    mock_registry.get.side_effect = AttributeError("test error")
    monkeypatch.setattr("ml_switcheroo_compiler.backends.mlx.eager.mlx_eager_registry", mock_registry)

    assert execute_op(DummyMlxBackend, "UnknownOp", np.array([1])) == "mock_numpy_fallback"


def test_mlx_eager_dtype_resolution(monkeypatch):
    """Test dtype resolution fallback logic."""

    mock_backend_module = MagicMock()
    # It shouldn't have 'unknown_dtype'
    del mock_backend_module.unknown_dtype
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    # _MLX_DTYPE_FALLBACK_MAP maps "float64" -> "float32"
    assert _mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"


def test_mlx_eager_topk_fallback(monkeypatch):
    """Test partition/topk fallback when return_indices is False and topk is not available."""

    mock_backend_module = MagicMock()
    # Ensure it doesn't have topk
    if hasattr(mock_backend_module, "topk"):
        del mock_backend_module.topk

    mock_backend_module.partition = MagicMock(return_value=np.array([[1, 2, 3]]))

    res = _mlx_partition(mock_backend_module, np.array([[3, 1, 2]]), k=2, return_indices=False)
    assert res is not None
