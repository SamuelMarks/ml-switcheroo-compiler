"""Tests for test_signal_coverage."""

from __future__ import annotations

from unittest.mock import patch

from ml_switcheroo_compiler.ops.signal.common_ops import _calculate_padding, _emit_signal_node


def test_signal_common_ops() -> None:
    """Verify signal common operations padding and node emission."""
    pad_cfg = _calculate_padding("same", "fill", 0.0)
    assert pad_cfg == {"mode": "same", "boundary": "fill", "fillvalue": 0.0}

    with patch("ml_switcheroo_compiler.ops.signal.common_ops._emit_linalg_node", return_value="mock_node") as mock_emit:
        res = _emit_signal_node("Convolve2D", [], {}, (3, 3), "float32")
        assert res == "mock_node"
        mock_emit.assert_called_once_with("Convolve2D", [], {}, [(3, 3)], ["float32"])
