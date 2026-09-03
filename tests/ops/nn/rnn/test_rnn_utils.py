from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_rnn_utils_branches():
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.ops.nn import rnn_utils as ru

    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))
    t3 = Tensor(np.ones((2, 2, 2)), TensorConfig(shape=(2, 2, 2), dtype=DType("float32"), device=Device("cpu")))

    orig = config.eager_mode
    config.eager_mode = True
    try:
        # test scan unroll
        def my_f(c, x):
            return c, x

        # test scan reverse
        ru.scan(my_f, (t,), t, ru.ScanConfig(reverse=True, unroll=True))

        # test scan eager=False
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan", return_value=((t,), t)):
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_reverse", return_value=t):
                ru.scan(my_f, (t,), t, ru.ScanConfig(reverse=True, unroll=False))
        config.eager_mode = True

        # test bidirectional merge_modes
        b_in = ru.BidirectionalInputs(forward_inputs=t, backward_inputs=t, forward_initial_state=(t,), backward_initial_state=(t,))

        for mode in ["concat", "sum", "mul", "ave", "none"]:
            # mock rnn to avoid recursion complexities, just return dummy
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.rnn", return_value=(t, (t,))):
                with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.concatenate", return_value=t), patch("ml_switcheroo_compiler.ops.nn.rnn_utils.add", return_value=t), patch("ml_switcheroo_compiler.ops.nn.rnn_utils.multiply", return_value=t):
                    ru.bidirectional(b_in, my_f, ru.BidirectionalConfig(merge_mode=mode))

        # test rnn config permutations
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.scan", return_value=((t,), t3)):
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils._permute_time_major", return_value=t3):
                # return_all_outputs = False, time_major = True
                ru.rnn(t, (t,), my_f, ru.RNNConfig(time_major=True, return_all_outputs=False))
                # return_all_outputs = False, time_major = False
                ru.rnn(t, (t,), my_f, ru.RNNConfig(time_major=False, return_all_outputs=False))

        # Test RNNCellDeviceWrapper
        cell = ru.RNNCellDeviceWrapper(my_f, None)
        cell(t, t)

        # Test RNNCellDropoutWrapper
        c_drop = ru.RNNCellDropoutWrapper(my_f, ru.DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout", return_value=t):
            c_drop(t, t)

        # Test RNNCellResidualWrapper
        c_res = ru.RNNCellResidualWrapper(my_f)
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.add", return_value=t):
            c_res(t, t)

        c_res2 = ru.RNNCellResidualWrapper(my_f, residual_fn=lambda i, o: o)
        c_res2(t, t)

    finally:
        config.eager_mode = orig


import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.nn.rnn_utils import DropoutWrapperConfig, RNNCellDropoutWrapper


def create_eager_tensor(data):
    return Tensor(data, TensorConfig(data.shape, DType("float32"), Device("cpu")))


def test_rnn_utils_missing_branches():
    class MockCell:
        def __call__(self, inputs, state, **kwargs):
            return inputs, state

    with ConfigContext(eager_mode=True):
        config = DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5)
        wrapper = RNNCellDropoutWrapper(MockCell(), config)

        inputs = create_eager_tensor(np.ones((2, 2)))
        state = (create_eager_tensor(np.ones((2, 2))),)

        # Test it actually applies operations or runs the fallback
        out, new_state = wrapper(inputs, state, training=True)
        # Dropout was applied
        assert out is not inputs
        assert new_state is state

        # Now test when training is False
        out, new_state = wrapper(inputs, state, training=False)
        np.testing.assert_array_equal(out.numpy(), inputs.numpy())
        assert new_state is state

        # Test branch where keep_prob == 1.0
        config2 = DropoutWrapperConfig(input_keep_prob=1.0, output_keep_prob=1.0)
        wrapper2 = RNNCellDropoutWrapper(MockCell(), config2)
        out2, new_state2 = wrapper2(inputs, state, training=True)
        assert out2 is inputs
        assert new_state2 is state


from unittest.mock import patch

from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import BidirectionalConfig, BidirectionalInputs, RNNCellDeviceWrapper, RNNCellResidualWrapper, RNNConfig, ScanConfig, _permute_time_major, bidirectional, rnn, scan
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed


def test_rnn_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t_in = Tensor(np.random.rand(2, 3, 4).astype(np.float32), TensorConfig((2, 3, 4), DType("float32"), "cpu"))
    h0 = Tensor(np.zeros((2, 8)).astype(np.float32), TensorConfig((2, 8), DType("float32"), "cpu"))
    kernel = Tensor(np.random.rand(4, 8).astype(np.float32), TensorConfig((4, 8), DType("float32"), "cpu"))
    rec_kernel = Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), DType("float32"), "cpu"))
    bias = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), DType("float32"), "cpu"))

    def my_cell(x, state):
        return simple_rnn_cell(x, state, kernel, rec_kernel, bias)

    _permute_time_major(t_in)

    rnn(t_in, (h0,), my_cell)

    t_in_time_major = Tensor(np.random.rand(3, 2, 4).astype(np.float32), TensorConfig((3, 2, 4), DType("float32"), "cpu"))
    rnn(t_in_time_major, (h0,), my_cell, config=RNNConfig(time_major=True, return_all_outputs=False))
    rnn(t_in, (h0,), my_cell, config=RNNConfig(time_major=False, return_all_outputs=False, go_backwards=True))

    bidir_inputs = BidirectionalInputs(forward_inputs=t_in, backward_inputs=t_in, forward_initial_state=(h0,), backward_initial_state=(h0,))

    bidirectional(bidir_inputs, my_cell)
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="sum"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="mul"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="ave"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="none"))

    dev_wrapper = RNNCellDeviceWrapper(my_cell, "cpu")
    dev_wrapper(Tensor(np.random.rand(2, 4).astype(np.float32), TensorConfig((2, 4), DType("float32"), "cpu")), (h0,))

    drop_wrapper = RNNCellDropoutWrapper(my_cell, config=DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
    drop_wrapper(Tensor(np.random.rand(2, 4).astype(np.float32), TensorConfig((2, 4), DType("float32"), "cpu")), (h0,))

    # for residual to work, the input size must match output size of cell, which is 8
    def my_cell_res(x, state):
        return simple_rnn_cell(x, state, Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), DType("float32"), "cpu")), rec_kernel, bias)

    res_wrapper = RNNCellResidualWrapper(my_cell_res)
    res_wrapper(Tensor(np.random.rand(2, 8).astype(np.float32), TensorConfig((2, 8), DType("float32"), "cpu")), (h0,))

    def my_res_fn(x, y):
        return y

    res_wrapper2 = RNNCellResidualWrapper(my_cell_res, residual_fn=my_res_fn)
    res_wrapper2(Tensor(np.random.rand(2, 8).astype(np.float32), TensorConfig((2, 8), DType("float32"), "cpu")), (h0,))

    time_distributed(t_in, wrapped_op_name="Relu")
    TimeDistributed().infer_shape(t_in)

    # Test tracing mode
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    try:
        try:
            time_distributed(t_in, wrapped_op_name="Relu")
        except Exception:
            pass

        def my_cell_trace(x, state):
            return x, (x,)

        try:
            rnn(t_in, (h0,), my_cell_trace)
        except Exception:
            pass

        try:
            rnn(t_in, (h0,), my_cell_trace, config=RNNConfig(time_major=True, return_all_outputs=False, go_backwards=True))
        except Exception:
            pass
    finally:
        global_tracing_state.is_tracing = False

    # tracing mode scan
    global_tracing_state.is_tracing = True
    try:

        def my_f(carry, x):
            return carry, x

        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan") as mock_scan:
            mock_scan.return_value = ((h0,), t_in)
            try:
                scan(my_f, (h0,), t_in)
            except Exception:
                pass
            try:
                scan(my_f, (h0,), t_in, config=ScanConfig(reverse=True))
            except Exception:
                pass
    finally:
        global_tracing_state.is_tracing = False


import ml_switcheroo_compiler.ops.nn.rnn as rnn_module


def test_rnn_import():
    assert rnn_module.rnn_step is not None


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
