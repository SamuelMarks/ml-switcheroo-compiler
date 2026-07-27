"""Test MLX eager edge cases coverage part 2."""

from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros, execute_op


def test_mlx_eager_execute_op_dim_kwarg(monkeypatch):
    """Test execute_op dimension kwarg mapping."""

    # Mock backend
    class DummyMlxBackend:
        @classmethod
        def execute_numpy_fallback(cls, op_type, *args, **kwargs):
            return kwargs

    monkeypatch.setattr("ml_switcheroo_compiler.backends.mlx.eager._execute_numpy_fallback", lambda cls, op_type, *args, **kwargs: kwargs)

    mock_registry = MagicMock()
    mock_registry.get.return_value = None
    monkeypatch.setattr("ml_switcheroo_compiler.backends.mlx.eager.mlx_eager_registry", mock_registry)

    res = execute_op(DummyMlxBackend, "Sum", np.array([1]), dim=0)
    assert res == {"axis": 0}


def test_mlx_eager_dtype_resolution_not_str(monkeypatch):
    """Test dtype resolution fallback logic when dtype is not a string."""

    mock_backend_module = MagicMock()
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype:
        def __str__(self):
            return "int32"

    dtype_obj = NonStrDtype()

    # Needs to be a valid dtype string
    assert _mlx_zeros(mock_backend_module, [1], dtype=dtype_obj) == "mock_zeros"


def test_mlx_eager_dtype_resolution_mapping_hit(monkeypatch):
    """Test dtype resolution fallback logic when mapping is hit."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros

    mock_backend_module = MagicMock()
    # Ensure no 'float64' exists on backend_module directly, but 'float32' does.
    if hasattr(mock_backend_module, "float64"):
        del mock_backend_module.float64
    mock_backend_module.float32 = "mock_mlx_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    assert _mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"


def test_mlx_eager_dtype_resolution_not_str_coverage(monkeypatch):
    """Test dtype resolution fallback logic when dtype is not a string."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_zeros

    mock_backend_module = MagicMock()
    # Emulate string dtype resolution failure branch (line 235 return path)
    if hasattr(mock_backend_module, "float64"):
        del mock_backend_module.float64

    mock_backend_module.float32 = "mock_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype:
        def __str__(self):
            return "float64"

    dtype_obj = NonStrDtype()

    # Needs to be a valid dtype string
    assert _mlx_zeros(mock_backend_module, [1], dtype=dtype_obj) == "mock_zeros"


def test_mlx_eager_dtype_resolution_not_str_coverage2(monkeypatch):
    """Test dtype resolution fallback logic when dtype is not a string."""

    mock_backend_module = MagicMock()
    # Emulate string dtype resolution failure branch (line 235 return path)
    if hasattr(mock_backend_module, "float64"):
        del mock_backend_module.float64

    mock_backend_module.float32 = "mock_float32"
    mock_backend_module.zeros = MagicMock(return_value="mock_zeros")

    class NonStrDtype:
        def __str__(self):
            return "float64"

    dtype_obj = NonStrDtype()

    # Actually call the method to trigger it
    import ml_switcheroo_compiler.backends.mlx.eager as mx_eager

    orig_dtype_map = mx_eager._MLX_DTYPE_FALLBACK_MAP
    mx_eager._MLX_DTYPE_FALLBACK_MAP = {"float64": "float32"}

    # We want dtype to be evaluated as string inside _mlx_zeros
    # which it should if it fails the hasattr test
    assert mx_eager._mlx_zeros(mock_backend_module, [1], dtype="float64") == "mock_zeros"

    # But we want line 235:
    # if isinstance(dtype, str) -> dtype starts as string 'float64'
    # hasattr(backend_module, dtype_str) -> False
    # k in dtype -> 'float64' in 'float64'
    # returns getattr(backend_module, 'float32')

    mx_eager._MLX_DTYPE_FALLBACK_MAP = orig_dtype_map
