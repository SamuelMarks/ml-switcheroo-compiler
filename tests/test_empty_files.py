import ml_switcheroo_compiler.ops.nn.rnn as rnn


def test_empty_files():
    assert rnn.rnn_step is not None
