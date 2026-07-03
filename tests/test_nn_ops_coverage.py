"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def setup_module() -> object:
    """Function docstring."""
    config.eager_mode = True


def test_conv1d() -> object:
    """Function docstring."""
    lhs = ops.array(np.random.randn(2, 10, 3).astype(np.float32))
    rhs = ops.array(np.random.randn(3, 3, 4).astype(np.float32))

    out1 = ops.conv1d(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv1d(lhs, rhs, strides=(2,), padding="SAME", lhs_dilation=2, rhs_dilation=2)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_conv2d() -> object:
    """Function docstring."""
    lhs = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    rhs = ops.array(np.random.randn(3, 3, 3, 4).astype(np.float32))

    out1 = ops.conv2d(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv2d(lhs, rhs, strides=(2, 2), padding="SAME", lhs_dilation=2, rhs_dilation=2)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_conv3d() -> object:
    """Function docstring."""
    lhs = ops.array(np.random.randn(2, 10, 10, 10, 3).astype(np.float32))
    rhs = ops.array(np.random.randn(3, 3, 3, 3, 4).astype(np.float32))

    out1 = ops.conv3d(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv3d(lhs, rhs, strides=(2, 2, 2), padding="SAME", lhs_dilation=2, rhs_dilation=2)
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_conv_transpose() -> object:
    """Function docstring."""
    lhs = ops.array(np.random.randn(2, 3, 10).astype(np.float32))
    rhs = ops.array(np.random.randn(4, 3, 3).astype(np.float32))

    out1 = ops.conv_transpose(lhs, rhs, strides=2, padding="SAME")
    out2 = ops.conv_transpose(lhs, rhs, strides=(2,), padding="SAME")
    assert out1 is not None
    assert out2 is not None

    out3 = ops.conv1d_transpose(lhs, rhs, strides=2, padding="SAME")
    assert out3 is not None

    lhs2d = ops.array(np.random.randn(2, 3, 10, 10).astype(np.float32))
    rhs2d = ops.array(np.random.randn(4, 3, 3, 3).astype(np.float32))
    out4 = ops.conv2d_transpose(lhs2d, rhs2d, strides=2, padding="SAME")
    assert out4 is not None

    lhs3d = ops.array(np.random.randn(2, 3, 10, 10, 10).astype(np.float32))
    rhs3d = ops.array(np.random.randn(4, 3, 3, 3, 3).astype(np.float32))
    out5 = ops.conv3d_transpose(lhs3d, rhs3d, strides=2, padding="SAME")
    assert out5 is not None

    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_pooling() -> object:
    """Function docstring."""
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


def test_rnn() -> object:
    """Function docstring."""
    inputs = ops.array(np.random.randn(10, 2, 5).astype(np.float32))
    initial_state = (ops.array(np.zeros((2, 4)).astype(np.float32)),)

    def cell_fn(x: object, state: object) -> object:
        """Function docstring."""
        return x, state

    out1, state1 = ops.rnn(inputs, initial_state, cell_fn, config=ops.RNNConfig(time_major=True))
    out2, state2 = ops.rnn(inputs, initial_state, cell_fn, config=ops.RNNConfig(time_major=False, go_backwards=True))
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_lstm_cell() -> object:
    """Function docstring."""
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


def test_gru_cell() -> object:
    """Function docstring."""
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


def test_activations() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 5).astype(np.float32))
    ops.softplus(x)
    ops.relu(x)
    ops.selu(x)
    ops.elu(x)
    ops.gelu(x)


def test_loss() -> object:
    """Function docstring."""
    y_true = ops.array(np.random.randn(2, 5).astype(np.float32))
    y_pred = ops.array(np.random.randn(2, 5).astype(np.float32))
    ops.dice_loss(y_true, y_pred)


def test_nlp() -> object:
    """Function docstring."""
    inputs = ops.array(np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).astype(np.int32))
    weights = ops.array(np.random.randn(10, 5).astype(np.float32))
    ops.embedding(inputs, weights)

    q = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    k = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    v = ops.array(np.random.randn(2, 3, 5).astype(np.float32))
    mask = ops.array(np.zeros((2, 3, 3)).astype(np.float32))

    ops.attention(ops.AttentionInputs(q, k, v))
    ops.attention(ops.AttentionInputs(q, k, v), ops.AttentionConfig(mask=mask, is_causal=True, dropout=0.1))


def test_pooling_explicit_padding() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    out1 = ops.max_pool(x, window_shape=(2, 2), padding=((1, 1), (1, 1)))
    out2 = ops.avg_pool(x, window_shape=(2, 2), padding=((1, 1), (1, 1)))
    assert out1 is not None
    assert out2 is not None
    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)


def test_depthwise_conv1d() -> object:
    """Function docstring."""
    lhs = Tensor(np.ones((2, 4, 3), dtype=np.float32), TensorConfig((2, 4, 3), "float32", "cpu"))
    rhs = Tensor(np.ones((2, 3, 2), dtype=np.float32), TensorConfig((2, 3, 2), "float32", "cpu"))
    config.eager_mode = True
    out1 = ops.depthwise_conv1d(lhs, rhs, strides=1, padding="VALID")
    assert out1.shape == (2, 3, 6)


def test_depthwise_conv2d() -> object:
    """Function docstring."""
    lhs = Tensor(np.ones((2, 4, 4, 3), dtype=np.float32), TensorConfig((2, 4, 4, 3), "float32", "cpu"))
    rhs = Tensor(np.ones((2, 2, 3, 2), dtype=np.float32), TensorConfig((2, 2, 3, 2), "float32", "cpu"))
    config.eager_mode = True
    out1 = ops.depthwise_conv2d(lhs, rhs, config=ops.GenericConvConfig(strides=1, padding="VALID"))
    assert out1.shape == (2, 3, 3, 6)


def test_average_pool() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    out = ops.average_pool(x, pool_size=(2, 2))
    assert out is not None


def test_batch_normalization() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    mean = ops.array(np.zeros((3,)).astype(np.float32))
    variance = ops.array(np.ones((3,)).astype(np.float32))
    offset = ops.array(np.zeros((3,)).astype(np.float32))
    scale = ops.array(np.ones((3,)).astype(np.float32))
    out = ops.batch_normalization(x, mean, variance, axis=-1, config=ops.BatchNormConfig(offset=offset, scale=scale))
    assert out is not None


def test_rms_normalization() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    scale = ops.array(np.ones((3,)).astype(np.float32))
    out = ops.rms_normalization(x, scale)
    assert out is not None


def test_generic_conv() -> object:
    """Function docstring."""
    lhs1 = ops.array(np.random.randn(2, 10, 3).astype(np.float32))
    rhs1 = ops.array(np.random.randn(3, 3, 4).astype(np.float32))
    out1 = ops.conv(lhs1, rhs1)

    lhs2 = ops.array(np.random.randn(2, 10, 10, 3).astype(np.float32))
    rhs2 = ops.array(np.random.randn(3, 3, 3, 4).astype(np.float32))
    out2 = ops.conv(lhs2, rhs2)

    lhs3 = ops.array(np.random.randn(2, 10, 10, 10, 3).astype(np.float32))
    rhs3 = ops.array(np.random.randn(3, 3, 3, 3, 4).astype(np.float32))
    out3 = ops.conv(lhs3, rhs3)

    assert out1 is not None and out2 is not None and out3 is not None


def test_generic_depthwise_conv() -> object:
    """Function docstring."""
    lhs1 = Tensor(np.ones((2, 4, 3), dtype=np.float32), TensorConfig((2, 4, 3), "float32", "cpu"))
    rhs1 = Tensor(np.ones((2, 3, 2), dtype=np.float32), TensorConfig((2, 3, 2), "float32", "cpu"))
    out1 = ops.depthwise_conv(lhs1, rhs1)

    lhs2 = Tensor(np.ones((2, 4, 4, 3), dtype=np.float32), TensorConfig((2, 4, 4, 3), "float32", "cpu"))
    rhs2 = Tensor(np.ones((2, 2, 3, 2), dtype=np.float32), TensorConfig((2, 2, 3, 2), "float32", "cpu"))
    out2 = ops.depthwise_conv(lhs2, rhs2)

    assert out1 is not None and out2 is not None


def test_generic_separable_conv() -> object:
    """Function docstring."""
    lhs1 = Tensor(np.ones((2, 4, 3), dtype=np.float32), TensorConfig((2, 4, 3), "float32", "cpu"))
    dw1 = Tensor(np.ones((2, 3, 2), dtype=np.float32), TensorConfig((2, 3, 2), "float32", "cpu"))
    pw1 = Tensor(np.ones((1, 6, 4), dtype=np.float32), TensorConfig((1, 6, 4), "float32", "cpu"))
    out1 = ops.separable_conv(lhs1, dw1, pw1)

    lhs2 = Tensor(np.ones((2, 4, 4, 3), dtype=np.float32), TensorConfig((2, 4, 4, 3), "float32", "cpu"))
    dw2 = Tensor(np.ones((2, 2, 3, 2), dtype=np.float32), TensorConfig((2, 2, 3, 2), "float32", "cpu"))
    pw2 = Tensor(np.ones((1, 1, 6, 4), dtype=np.float32), TensorConfig((1, 1, 6, 4), "float32", "cpu"))
    out2 = ops.separable_conv(lhs2, dw2, pw2)

    assert out1 is not None and out2 is not None


def test_new_activations() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 6).astype(np.float32))
    ops.celu(x)
    ops.glu(x)
    ops.hard_shrink(x)
    ops.hard_sigmoid(x)
    ops.hard_silu(x)
    ops.hard_swish(x)
    ops.hard_tanh(x)
    ops.leaky_relu(x)
    ops.log_sigmoid(x)
    ops.log_softmax(x)
    ops.relu6(x)
    ops.sigmoid(x)
    ops.silu(x)
    ops.soft_shrink(x)
    ops.softmax(x)
    ops.softsign(x)
    ops.sparse_plus(x)
    ops.sparse_sigmoid(x)
    ops.sparsemax(x)
    ops.squareplus(x)
    ops.swish(x)
    ops.tanh_shrink(x)
    ops.threshold(x)


def test_new_losses() -> object:
    """Function docstring."""
    y_true = ops.array(np.array([[1.0, 0.0], [0.0, 1.0]]).astype(np.float32))
    y_pred = ops.array(np.array([[0.9, 0.1], [0.2, 0.8]]).astype(np.float32))

    bce = ops.binary_crossentropy(y_true, y_pred)
    assert bce is not None

    cce = ops.categorical_crossentropy(y_true, y_pred)
    assert cce is not None

    y_true_sparse = ops.array(np.array([0, 1]).astype(np.int32))
    scce = ops.sparse_categorical_crossentropy(y_true_sparse, y_pred)
    assert scce is not None

    inputs = ops.array(np.random.randn(2, 5, 3).astype(np.float32))
    lengths = ops.array(np.array([5, 4]).astype(np.int32))
    paths, log_probs = ops.ctc_decode(inputs, lengths)
    assert paths is not None


def test_utility_ops() -> object:
    """Function docstring."""
    config.eager_mode = True

    # dot_product_attention
    q = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    k = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    v = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    out = ops.dot_product_attention(q, k, v)
    assert out is not None

    # psnr
    a = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    b = ops.array(np.random.randn(2, 3, 4).astype(np.float32))
    p = ops.psnr(a, b, max_val=1.0)
    assert p is not None

    # in_top_k
    targets = ops.array(np.array([1, 2]).astype(np.int32))
    predictions = ops.array(np.array([[0.1, 0.9, 0.2], [0.1, 0.2, 0.9]]).astype(np.float32))
    topk = ops.in_top_k(targets, predictions, k=1)
    assert topk is not None

    # check image alias
    assert ops.image is not None


def test_activity_regularization() -> object:
    """Function docstring."""
    x = ops.array(np.random.randn(2, 2).astype(np.float32))
    _ = ops.activity_regularization(x, l1=0.1, l2=0.2)
    pass


def test_activity_regularization_ast() -> object:
    """Function docstring."""
    config.eager_mode = False

    global_tracing_state.start_tracing("Test")
    x = ops.array(np.array([1.0, 2.0]))
    _ = ops.activity_regularization(x, l1=0.1, l2=0.2)
    graph = global_tracing_state.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()

    assert code_str is not None


def test_adaptive_avg_pool2d() -> object:
    """Function docstring."""
    global_tracing_state.start_tracing("Test")
    x = ops.array(np.random.randn(2, 3, 4, 4).astype(np.float32))

    _ = ops.adaptive_avg_pool2d(x, output_size=(2, 2))
    global_tracing_state.stop_tracing()
    pass


def test_adaptive_max_pool2d() -> object:
    """Function docstring."""
    global_tracing_state.start_tracing("Test")
    x = ops.array(np.random.randn(2, 3, 4, 4).astype(np.float32))

    _ = ops.adaptive_max_pool2d(x, output_size=(2, 2))
    global_tracing_state.stop_tracing()
    pass


def test_adaptive_avg_pool2d_ast() -> object:
    """Function docstring."""
    config.eager_mode = False
    global_tracing_state.start_tracing("Test")
    x = ops.array(np.array([[[[1.0]]]]))
    _ = ops.adaptive_avg_pool2d(x, output_size=(2, 2))
    graph = global_tracing_state.stop_tracing()
    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_adaptive_avg_pool2d" in code_str


def test_adaptive_max_pool2d_ast() -> object:
    """Function docstring."""
    config.eager_mode = False
    global_tracing_state.start_tracing("Test")
    x = ops.array(np.array([[[[1.0]]]]))
    _ = ops.adaptive_max_pool2d(x, output_size=(2, 2))
    graph = global_tracing_state.stop_tracing()
    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_adaptive_max_pool2d" in code_str
