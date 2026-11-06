"""Tests for RNN initialization and registration."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.nn.rnn as rnn
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY
from tests.testing_utils import has_backend_member


def test_rnn_operations_exist_and_registered() -> None:
    """Verify that RNN operations are properly initialized and registered."""
    assert rnn.rnn_step is not None

    # Check that RNN-related ops are in the global registry
    assert "Rnn" in _OP_REGISTRY or "RnnStep" in _OP_REGISTRY or "Lstm" in _OP_REGISTRY or "Gru" in _OP_REGISTRY or "RnnCell" in _OP_REGISTRY

    # Check that the module exposes expected attributes
    assert has_backend_member(rnn, "rnn") or has_backend_member(rnn, "rnn_step")
