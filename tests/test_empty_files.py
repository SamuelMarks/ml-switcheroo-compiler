import ml_switcheroo_compiler.ops.nn.rnn as rnn
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY


def test_rnn_operations_exist_and_registered():
    """Verify that RNN operations are properly initialized and registered."""
    assert rnn.rnn_step is not None

    # Check that RNN-related ops are in the global registry
    assert "Rnn" in _OP_REGISTRY or "RnnStep" in _OP_REGISTRY or "Lstm" in _OP_REGISTRY or "Gru" in _OP_REGISTRY or "RnnCell" in _OP_REGISTRY

    # Check that the module exposes expected attributes
    assert hasattr(rnn, "rnn") or hasattr(rnn, "rnn_step")
