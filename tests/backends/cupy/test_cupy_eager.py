"""Unit tests for native CuPy eager execution, device residency, and dispatch."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.cupy.eager import _verify_gpu_residency, execute_op
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_cupy_verify_gpu_residency() -> None:
    """Test verification of GPU device residency for CuPy ndarrays."""

    class MockNdarray:
        def __init__(self, data=None):
            self.data = data

    class DataWithDevice:
        def __init__(self, device=None):
            self.device = device

    class DataNoDevice:
        pass

    # Mock cupy array resident on GPU when cp is None
    mock_array = MockNdarray(DataWithDevice("cuda:0"))
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", None):
        _verify_gpu_residency(mock_array)

    # Valid device memory with cp not None
    mock_cp = MagicMock()
    mock_cp.ndarray = MockNdarray
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
        _verify_gpu_residency(mock_array)

        # res has data but data has no device attribute
        _verify_gpu_residency(MockNdarray(DataNoDevice()))

        # res has data = None
        _verify_gpu_residency(MockNdarray(None))

        # res is not an array
        _verify_gpu_residency(42)

    # Mock array with missing device (device is None)
    bad_array = MockNdarray(DataWithDevice(None))
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
        with pytest.raises(RuntimeError, match="CuPy array is not residing on GPU device memory"):
            _verify_gpu_residency(bad_array)


def test_cupy_eager_dispatch_native() -> None:
    """Test dedicated eager dispatch executing natively on CuPy without CPU fallback."""
    mock_cp = MagicMock()
    mock_cp.add.return_value = MagicMock()
    mock_cp.add.return_value.data.device = "cuda:0"

    with patch.dict(sys.modules, {"cupy": mock_cp}):
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
            res = execute_op(None, "Add", [1.0, 2.0], [3.0, 4.0])
            assert res == mock_cp.add.return_value
            mock_cp.add.assert_called_once()

            # Non-list arg (e.g. scalar int) and asarray exception handling
            mock_cp.asarray.side_effect = RuntimeError("asarray failure")
            res_scalar = execute_op(None, "Add", [1.0, 2.0], 5)
            assert res_scalar == mock_cp.add.return_value
            mock_cp.asarray.side_effect = None

            # Ensure silent CPU NumPy fallback is prevented: unmapped op raises BackendNotSupportedError
            with pytest.raises(BackendNotSupportedError):
                execute_op(None, "CompletelyUnsupportedOp123")

            # Dispatch raises BackendNotSupportedError
            with patch("ml_switcheroo_compiler.backends.mapping_loader.dispatch_eager_op", side_effect=BackendNotSupportedError("dispatch failed")):
                with pytest.raises(BackendNotSupportedError, match="Operation 'Add' is not implemented"):
                    execute_op(None, "Add", [1.0])


def test_cupy_eager_unavailable() -> None:
    """Test that BackendNotSupportedError is raised when CuPy is unavailable."""
    with patch.dict(sys.modules, {"cupy": None}):
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", None):
            with pytest.raises(BackendNotSupportedError, match="CuPy is not installed"):
                execute_op(None, "Add", 1, 2)


def test_cupy_eager_reductions_and_kwarg_mapping() -> None:
    """Test keyword argument translation (dim->axis, keepdim->keepdims) on CuPy."""
    mock_cp = MagicMock()
    mock_cp.sum.return_value = MagicMock()
    mock_cp.sum.return_value.data.device = "cuda:0"

    with patch.dict(sys.modules, {"cupy": mock_cp}):
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
            execute_op(None, "Sum", [1.0, 2.0], dim=0, keepdim=True)
            mock_cp.sum.assert_called_once()
            _, kwargs = mock_cp.sum.call_args
            assert "axis" in kwargs and kwargs["axis"] == 0
            assert "keepdims" in kwargs and kwargs["keepdims"] is True


def test_cupy_eager_fallback_mock_mod() -> None:
    """Test CuPy eager fallback when target_name is present on current module (line 69)."""
    import ml_switcheroo_compiler.backends.cupy.eager as cupy_eager

    cupy_eager.add = lambda a, b: 42
    try:
        with patch.dict(sys.modules, {"cupy": None}):
            with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", None):
                res = execute_op(None, "Add", 1, 2)
                assert res == 42
    finally:
        delattr(cupy_eager, "add")


def test_cupy_eager_registry_fallback() -> None:
    """Test CuPy eager dispatch fallback to global_eager_registry (line 94)."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    mock_func = MagicMock(return_value="registered_res")
    with patch.object(global_eager_registry, "get", return_value=mock_func):
        res = execute_op(None, "CustomRegisteredOp", 10)
        assert res == "registered_res"
        mock_func.assert_called_once()


def test_cupy_eager_remaining_branches() -> None:
    """Test lines 63, 86, 167, and 207 in cupy eager.py."""
    from ml_switcheroo_compiler.backends.cupy.eager import (
        _attach_vision_ops,
        cupy_conv2d,
        cupy_resize_bilinear,
    )

    mock_cp = MagicMock()

    # 1. Line 63: 4D conv2d with pad_h == 0 and pad_w == 0
    mock_x4d = MagicMock()
    mock_x4d.ndim = 4
    mock_x4d.shape = (1, 1, 3, 3)
    mock_x4d.dtype = "float32"
    mock_w = MagicMock()
    mock_w.ndim = 4
    mock_w.shape = (1, 1, 1, 1)

    mock_cp.zeros.return_value = MagicMock()
    mock_cp.tensordot.return_value = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
        out_conv = cupy_conv2d(mock_x4d, mock_w, stride=1, padding="VALID")
        assert out_conv is not None
        mock_cp.pad.assert_not_called()

    # 2. Line 86: conv2d with non-2D/4D input (e.g. 1D)
    mock_x1d = MagicMock()
    mock_x1d.ndim = 1
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
        assert cupy_conv2d(mock_x1d, mock_w) is mock_x1d

    # 3. Line 167: resize_bilinear with same shape (new_h, new_w) == (h, w)
    mock_x_img = MagicMock()
    mock_x_img.ndim = 4
    mock_x_img.shape = (1, 3, 8, 8)
    mock_x_img.copy.return_value = "copied_img"
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
        res_resize = cupy_resize_bilinear(mock_x_img, size=(8, 8))
        assert res_resize == "copied_img"

    # 4. Line 207: _attach_vision_ops when attributes are missing
    class EmptyBackendMod:
        pass

    empty_mod = EmptyBackendMod()
    _attach_vision_ops(empty_mod)
    assert hasattr(empty_mod, "conv2d")
    assert hasattr(empty_mod, "maxpool2d")
    assert hasattr(empty_mod, "avgpool2d")
    assert hasattr(empty_mod, "resize_bilinear")
