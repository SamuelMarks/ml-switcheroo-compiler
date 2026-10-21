"""Tests for test_binary_init."""

from __future__ import annotations

import typing

from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_binary_init_funcs() -> None:
    """Test function test_binary_init_funcs."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary as binary

    with patch("ml_switcheroo_compiler.ops.creation.zeros_like", return_value="zeros"):
        with patch("ml_switcheroo_compiler.ops.shape.indexing.where", return_value="where"):
            with patch("ml_switcheroo_compiler.ops.binary.divide", return_value="divide"):
                with patch("ml_switcheroo_compiler.ops.binary.equal", return_value="equal"):
                    res = binary.divide_no_nan(1, 2)
                    assert res == "where"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = "res"
        assert binary.polar(1, 2) == "res"
        assert binary.view_as_complex(1) == "res"
        assert binary.view_as_real(1) == "res"


def test_binary_init_missing() -> None:
    """Test function test_binary_init_missing."""
    import importlib
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary

    with patch("ml_switcheroo_compiler.ops.base.get_op", side_effect=KeyError("mock")):
        importlib.reload(ml_switcheroo_compiler.ops.binary)
        assert as_tensor(ml_switcheroo_compiler.ops.binary.add) is None

    importlib.reload(ml_switcheroo_compiler.ops.binary)
