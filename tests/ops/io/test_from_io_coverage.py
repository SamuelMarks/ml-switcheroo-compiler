"""Tests for test_from_io_coverage."""

from __future__ import annotations

from unittest.mock import patch

import ml_switcheroo_compiler.ops.io.from_io as from_io_mod


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_from_io_full_coverage() -> None:
    """Test full coverage for from_io op definitions and functional dispatchers."""
    ff_op = from_io_mod.Fromfile()
    assert ff_op.infer_shape(count=-1) == (None,)
    assert ff_op.infer_shape(count=10) == (10,)

    fs_op = from_io_mod.Fromstring()
    assert fs_op.infer_shape(count=-1) == (None,)
    assert fs_op.infer_shape(count=5) == (5,)

    fi_op = from_io_mod.Fromiter()
    assert fi_op.infer_shape(count=-1) == (None,)
    assert fi_op.infer_shape(count=8) == (8,)

    fn_op = from_io_mod.Fromfunction()
    assert fn_op.infer_shape(None, shape=(3, 4)) == (3, 4)
    assert fn_op.infer_shape(None, shape=5) == (5,)
    assert fn_op.infer_shape(None, [2, 2]) == (2, 2)
    assert fn_op.infer_shape(None) == ()

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromfile") as mock_disp:
        assert from_io_mod.fromfile("file.bin", dtype=float, count=2) == "dispatched_fromfile"
        mock_disp.assert_called_once_with("Fromfile", "file.bin", dtype=float, count=2, sep="", offset=0, like=None)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromstring") as mock_disp:
        assert from_io_mod.fromstring("1 2", dtype=float, count=2) == "dispatched_fromstring"
        mock_disp.assert_called_once_with("Fromstring", "1 2", dtype=float, count=2, sep="", like=None)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromiter") as mock_disp:
        assert from_io_mod.fromiter([1, 2], dtype=float, count=2) == "dispatched_fromiter"
        mock_disp.assert_called_once_with("Fromiter", [1, 2], float, count=2, like=None)

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="dispatched_fromfn") as mock_disp:
        assert from_io_mod.fromfunction(lambda i: i, (3,)) == "dispatched_fromfn"
        mock_disp.assert_called_once()
