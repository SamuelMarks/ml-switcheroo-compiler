from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
import numpy as np


def setup_module():
    config.eager_mode = True


def test_conv1d():
    lhs = ops.array(np.random.randn(2, 10, 3).astype(np.float32))
    rhs = ops.array(np.random.randn(3, 3, 4).astype(np.float32))

    out1 = ops.conv1d(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv1d(lhs, rhs, strides=(2,), padding="SAME", lhs_dilation=2, rhs_dilation=2)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_conv2d():
    lhs = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    rhs = ops.array(np.random.randn(3, 3, 3, 4).astype(np.float32))

    out1 = ops.conv2d(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv2d(lhs, rhs, strides=(2, 2), padding="SAME", lhs_dilation=2, rhs_dilation=2)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_conv3d():
    lhs = ops.array(np.random.randn(2, 10, 10, 10, 3).astype(np.float32))
    rhs = ops.array(np.random.randn(3, 3, 3, 3, 4).astype(np.float32))

    out1 = ops.conv3d(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv3d(lhs, rhs, strides=(2, 2, 2), padding="SAME", lhs_dilation=2, rhs_dilation=2)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_conv_transpose():
    lhs = ops.array(np.random.randn(2, 3, 10).astype(np.float32))
    rhs = ops.array(np.random.randn(4, 3, 3).astype(np.float32))

    out1 = ops.conv_transpose(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv_transpose(lhs, rhs, strides=(2,), padding="SAME")
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_pooling():
    x = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    out1 = ops.max_pool(x, window_shape=(2, 2))
    out2 = ops.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    out3 = ops.avg_pool(x, window_shape=(2, 2))
    assert out3 is not None
    out4 = ops.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    assert out4 is not None
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_rnn():
    inputs = ops.array(np.random.randn(10, 2, 5).astype(np.float32))
    initial_state = (ops.array(np.zeros((2, 4)).astype(np.float32)),)

    def cell_fn(x, state):
        return x, state

    out1, state1 = ops.rnn(inputs, initial_state, cell_fn, time_major=True)
    out2, state2 = ops.rnn(inputs, initial_state, cell_fn, time_major=False, go_backwards=True)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_lstm_cell():
    inputs = ops.array(np.random.randn(2, 5).astype(np.float32))
    state = (
        ops.array(np.zeros((2, 4)).astype(np.float32)),
        ops.array(np.zeros((2, 4)).astype(np.float32)),
    )
    kernel = ops.array(np.random.randn(5, 16).astype(np.float32))
    recurrent_kernel = ops.array(np.random.randn(4, 16).astype(np.float32))
    bias = ops.array(np.random.randn(16).astype(np.float32))

    out1, state1 = ops.lstm_cell(inputs, state, kernel, recurrent_kernel, bias)
    out2, state2 = ops.lstm_cell(inputs, state, kernel, recurrent_kernel)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_gru_cell():
    inputs = ops.array(np.random.randn(2, 5).astype(np.float32))
    state = ops.array(np.zeros((2, 4)).astype(np.float32))
    kernel = ops.array(np.random.randn(5, 12).astype(np.float32))
    recurrent_kernel = ops.array(np.random.randn(4, 12).astype(np.float32))
    bias = ops.array(np.random.randn(12).astype(np.float32))

    out1, state1 = ops.gru_cell(inputs, state, kernel, recurrent_kernel, bias)
    out2, state2 = ops.gru_cell(inputs, state, kernel, recurrent_kernel)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_activations():
    x = ops.array(np.random.randn(2, 5).astype(np.float32))
    ops.softplus(x)
    ops.relu(x)
    ops.selu(x)
    ops.elu(x)
    ops.gelu(x)


def test_loss():
    y_true = ops.array(np.random.randn(2, 5).astype(np.float32))
    y_pred = ops.array(np.random.randn(2, 5).astype(np.float32))
    ops.dice_loss(y_true, y_pred)


def test_nlp():
    inputs = ops.array(np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).astype(np.int32))
    weights = ops.array(np.random.randn(10, 5).astype(np.float32))
    ops.embedding(inputs, weights)

    q = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    k = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    v = ops.array(np.random.randn(2, 3, 5).astype(np.float32))
    mask = ops.array(np.zeros((2, 3, 3)).astype(np.float32))

    ops.attention(q, k, v)
    ops.attention(q, k, v, mask=mask, is_causal=True, dropout=0.1)


def test_pooling_explicit_padding():
    x = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    out1 = ops.max_pool(x, window_shape=(2, 2), padding=((1, 1), (1, 1)))
    out2 = ops.avg_pool(x, window_shape=(2, 2), padding=((1, 1), (1, 1)))
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)
