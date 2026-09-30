"""Tests for test_file_io_coverage."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.io.file_io as file_io_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_file_io_coverage(tmp_path: pytest.TempPathFactory) -> None:
    """Verify 100% line and branch coverage for file_io operations.

    Args:
        tmp_path (pytest.TempPathFactory): Pytest fixture for temporary directory.
    """
    orig_eager = config.eager_mode
    try:
        dummy_t = Tensor(np.array(["test"]), TensorConfig((1,), DType.String, Device("cpu")))
        # Eager mode
        config.eager_mode = True
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = "file_op_res"

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
            assert file_io_mod.read_file("file.txt", name="r1") == "file_op_res"
            mock_backend.execute_op.assert_called_with("ReadFile", "file.txt", name="r1")

            assert file_io_mod.write_file("file.txt", dummy_t, name="w1") == "file_op_res"
            mock_backend.execute_op.assert_called_with("WriteFile", "file.txt", dummy_t, name="w1")

        # Non-eager mode
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", side_effect=lambda op, *args, **kwargs: f"emitted_{op}"):
            assert file_io_mod.read_file("file.txt") == "emitted_ReadFile"
            assert file_io_mod.write_file("file.txt", dummy_t) == "emitted_WriteFile"

        # OpDefs infer_shape
        rf_op = file_io_mod.ReadFile()
        assert rf_op.infer_shape() == ()
        assert rf_op.infer_shape(DummyWithShape((2, 1)), DummyWithShape((1, 3))) == (2, 3)

        wf_op = file_io_mod.WriteFile()
        assert wf_op.infer_shape() == ()
        assert wf_op.infer_shape(DummyWithShape((4,))) == (4,)
        assert wf_op.infer_shape(DummyWithShape((4, 1)), DummyWithShape((1, 5))) == (4, 5)

        # Local gfile helper functions
        temp_dir = str(tmp_path)
        sub_dir = os.path.join(temp_dir, "test_dir", "sub")
        file_io_mod.gfile_makedirs(sub_dir)
        assert os.path.exists(sub_dir)

        src_file = os.path.join(sub_dir, "source.txt")
        dst_file = os.path.join(sub_dir, "dest.txt")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write("test content")

        # gfile_copy without overwrite
        file_io_mod.gfile_copy(src_file, dst_file, overwrite=False)
        assert os.path.exists(dst_file)

        # gfile_copy with overwrite=False when destination exists raises FileExistsError
        with pytest.raises(FileExistsError, match="already exists"):
            file_io_mod.gfile_copy(src_file, dst_file, overwrite=False)

        # gfile_copy with overwrite=True
        file_io_mod.gfile_copy(src_file, dst_file, overwrite=True)

        # gfile_glob
        glob_res = file_io_mod.gfile_glob(os.path.join(sub_dir, "*.txt"))
        assert len(glob_res) == 2

        # gfile_stat
        stat_res = file_io_mod.gfile_stat(src_file)
        assert "length" in stat_res
        assert "mtime" in stat_res
        assert stat_res["length"] == len("test content")
    finally:
        config.eager_mode = orig_eager
