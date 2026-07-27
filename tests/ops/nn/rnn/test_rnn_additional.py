import ml_switcheroo_compiler.ops.nn.rnn as rnn


def test_rnn_import():
    assert rnn.rnn_step is not None


def test_rnn_cell_coverage():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell

    orig_eager = config.eager_mode
    config.eager_mode = True

    # dummy test since eager mode handles matmul and add natively
    inputs = Tensor(np.array([[1.0, 2.0]]), TensorConfig((1, 2), "float32", "cpu"))
    state = (Tensor(np.array([[0.5]]), TensorConfig((1, 1), "float32", "cpu")),)
    kernel = Tensor(np.array([[1.0], [2.0]]), TensorConfig((2, 1), "float32", "cpu"))
    recurrent_kernel = Tensor(np.array([[0.1]]), TensorConfig((1, 1), "float32", "cpu"))
    bias = Tensor(np.array([0.1]), TensorConfig((1,), "float32", "cpu"))

    try:
        out, next_st = simple_rnn_cell(inputs, state, kernel, recurrent_kernel, bias)
        assert out.shape == (1, 1)

        # Test without bias
        out, next_st = simple_rnn_cell(inputs, state, kernel, recurrent_kernel, None)
        assert out.shape == (1, 1)
    finally:
        config.eager_mode = orig_eager


def test_rnn_utils_coverage():
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.rnn_utils as rnn_utils
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None

    try:
        t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))

        # Test scan eager unrolled
        config.eager_mode = True

        def mock_cell(carry, x):
            return carry, x

        carry, ys = rnn_utils.scan(mock_cell, t, t, config=rnn_utils.ScanConfig(unroll=True, reverse=True))
        assert ys.shape == (2, 1, 2)

        # Test bidirectional
        config.eager_mode = False
        inputs = rnn_utils.BidirectionalInputs(forward_inputs=t, backward_inputs=t, forward_initial_state=(t,), backward_initial_state=(t,))

        from unittest.mock import patch

        def mock_rnn(inputs, state, cell_fn, config):
            return inputs, state

        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.rnn", mock_rnn):
            rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="concat"))
            rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="sum"))
            rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="mul"))
            rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="ave"))
            rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode=None))

        # Test rnn
        def mock_scan(fn, init, xs, config):
            c, out = fn(init, xs)  # trigger the scan_fn

            class DummyOut:
                shape = (1, 1, 1)

                def __getitem__(self, key):
                    return self

            return c, DummyOut()

        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.scan", mock_scan):
            # Non time major, return all
            rnn_utils.rnn(t, (t,), mock_cell, config=rnn_utils.RNNConfig(time_major=False, return_all_outputs=True))
            # Time major, return last
            rnn_utils.rnn(t, (t,), mock_cell, config=rnn_utils.RNNConfig(time_major=True, return_all_outputs=False))

        # Test DeviceWrapper
        class MockCell:
            def __call__(self, x, state):
                return x, state

        w = rnn_utils.RNNCellDeviceWrapper(MockCell(), "cpu")
        w(t, (t,))

        # Test ResidualWrapper
        rw = rnn_utils.RNNCellResidualWrapper(MockCell())
        rw(t, (t,))

        rw2 = rnn_utils.RNNCellResidualWrapper(MockCell(), residual_fn=lambda i, o: i)
        rw2(t, (t,))

        # Test DropoutWrapper
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout", return_value=t):
            dw = rnn_utils.RNNCellDropoutWrapper(MockCell(), config=rnn_utils.DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
            dw(t, (t,))

    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_rnn_utils_coverage2():
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.rnn_utils as rnn_utils
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None

    try:
        t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))

        def mock_cell(carry, x):
            return carry, x

        # Test scan NOT eager, unroll=False
        from unittest.mock import patch

        def mock_cf_scan(f, init, xs, length):
            return init, xs

        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan", mock_cf_scan):
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_reverse", return_value=t):
                carry, ys = rnn_utils.scan(mock_cell, t, t, config=rnn_utils.ScanConfig(unroll=False, reverse=True))

    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
