"""Tests for test_rnn_exports."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.nn.rnn as rnn_module
import ml_switcheroo_compiler.ops.nn.rnn_cell as rnn_cell_module


def test_rnn_exports() -> None:
    """Test RNN module exports and aliased step definitions."""
    assert rnn_module.rnn_step is rnn_cell_module.simple_rnn_cell
    assert rnn_module.lstm_step is rnn_module.lstm_cell
    assert rnn_module.gru_step is rnn_module.gru_cell

    for item in rnn_module.__all__:
        assert hasattr(rnn_module, item)
