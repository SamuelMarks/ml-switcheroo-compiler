"""Unit tests for native Dask eager execution and dispatch."""

import builtins
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo_compiler.backends.dask.eager as dask_eager
from ml_switcheroo_compiler.backends.dask.eager import execute_op
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_dask_eager_dispatch_native() -> None:
    """Test dedicated eager dispatch executing natively on Dask without CPU fallback."""
    mock_da = MagicMock()
    mock_da.add.return_value = MagicMock()

    with patch.dict(sys.modules, {"dask.array": mock_da}):
        with patch("ml_switcheroo_compiler.backends.dask.eager.da", mock_da):
            res = execute_op(None, "Add", [1.0, 2.0], [3.0, 4.0])
            assert res == mock_da.add.return_value
            mock_da.add.assert_called_once()

            # Non-list arg (e.g. float) to test branch 60->66 and line 66
            res_scalar = execute_op(None, "Add", 1.0, 2.0)
            assert res_scalar == mock_da.add.return_value

            # from_array raises Exception to test lines 63-64
            mock_da.from_array.side_effect = RuntimeError("from_array error")
            res_err = execute_op(None, "Add", [1.0, 2.0])
            assert res_err == mock_da.add.return_value
            mock_da.from_array.side_effect = None

            # Ensure silent CPU NumPy fallback is prevented: unmapped op raises BackendNotSupportedError
            with pytest.raises(BackendNotSupportedError):
                execute_op(None, "NonExistentDaskOp")

            # dispatch_eager_op raises BackendNotSupportedError
            with patch("ml_switcheroo_compiler.backends.mapping_loader.dispatch_eager_op", side_effect=BackendNotSupportedError("dispatch failure")):
                with pytest.raises(BackendNotSupportedError, match="Operation 'Add' is not implemented"):
                    execute_op(None, "Add", 1.0)


def test_dask_import_error() -> None:
    """Test handling of ImportError when dask.array is not available at import time."""
    orig_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if "dask" in name:
            raise ImportError("Mocked missing dask")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(dask_eager)
        assert dask_eager.da is None

    # Restore module
    importlib.reload(dask_eager)
    assert dask_eager.da is not None


def test_dask_eager_unavailable() -> None:
    """Test that BackendNotSupportedError is raised when Dask is unavailable."""
    with patch("ml_switcheroo_compiler.backends.dask.eager.da", None):
        with pytest.raises(BackendNotSupportedError, match="Dask is not installed"):
            execute_op(None, "Add", 1, 2)


def test_dask_eager_reductions_and_kwarg_mapping() -> None:
    """Test keyword argument translation (dim->axis, keepdim->keepdims) on Dask."""
    mock_da = MagicMock()
    mock_da.sum.return_value = MagicMock()

    with patch.dict(sys.modules, {"dask.array": mock_da}):
        with patch("ml_switcheroo_compiler.backends.dask.eager.da", mock_da):
            execute_op(None, "Sum", [1.0, 2.0], dim=0, keepdim=True)
            mock_da.sum.assert_called_once()
            _, kwargs = mock_da.sum.call_args
            assert "axis" in kwargs and kwargs["axis"] == 0
            assert "keepdims" in kwargs and kwargs["keepdims"] is True


def test_dask_eager_mock_entry_backend_mod() -> None:
    """Test Dask eager with non-ModuleType in sys.modules (line 38)."""
    mock_entry = MagicMock()
    mock_entry.add.return_value = 999
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.dask.eager": mock_entry}):
        res = execute_op(None, "Add", 1, 2)
        assert res == 999


def test_dask_eager_registry_fallback() -> None:
    """Test Dask eager fallback to global_eager_registry (line 69)."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    mock_func = MagicMock(return_value="dask_reg_res")
    with patch.object(global_eager_registry, "get", return_value=mock_func):
        res = execute_op(None, "CustomDaskRegOp", 5)
        assert res == "dask_reg_res"


def test_dask_eager_remaining_branches() -> None:
    """Test lines 26, 51, and 79 in dask eager.py."""
    from ml_switcheroo_compiler.backends.dask.eager import (
        _verify_dask_task_graph,
        dask_conv2d,
        propagate_chunk_shapes,
    )

    # 1. Line 26: _verify_dask_task_graph when da is None
    with patch("ml_switcheroo_compiler.backends.dask.eager.da", None):
        assert _verify_dask_task_graph("anything") is None

    # 2. Line 51: propagate_chunk_shapes when da is None
    with patch("ml_switcheroo_compiler.backends.dask.eager.da", None):
        assert propagate_chunk_shapes([1, 2, 3]) == ()

    # 3. Line 79: dask_conv2d with padding="VALID" and 1x1 kernel (pad_h == 0 and pad_w == 0)
    mock_da = MagicMock()
    mock_x = MagicMock()
    mock_x.ndim = 4
    mock_x.shape = (1, 1, 3, 3)
    mock_x.dtype = "float32"
    mock_x.chunks = ((1,), (1,), (3,), (3,))
    mock_w = MagicMock()
    mock_w.ndim = 4
    mock_w.shape = (1, 1, 1, 1)

    mock_da.zeros.return_value = MagicMock()
    with patch("ml_switcheroo_compiler.backends.dask.eager.da", mock_da):
        out = dask_conv2d(mock_x, mock_w, stride=1, padding="VALID")
        assert out is not None
        mock_da.pad.assert_not_called()
