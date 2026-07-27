from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import BidirectionalConfig, BidirectionalInputs, DropoutWrapperConfig, RNNCellDeviceWrapper, RNNCellDropoutWrapper, RNNCellResidualWrapper, RNNConfig, ScanConfig, _permute_time_major, bidirectional, rnn, scan
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed


def test_rnn_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t_in = Tensor(np.random.rand(2, 3, 4).astype(np.float32), TensorConfig((2, 3, 4), "float32", "cpu"))
    h0 = Tensor(np.zeros((2, 8)).astype(np.float32), TensorConfig((2, 8), "float32", "cpu"))
    kernel = Tensor(np.random.rand(4, 8).astype(np.float32), TensorConfig((4, 8), "float32", "cpu"))
    rec_kernel = Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), "float32", "cpu"))
    bias = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), "float32", "cpu"))

    def my_cell(x, state):
        return simple_rnn_cell(x, state, kernel, rec_kernel, bias)

    _permute_time_major(t_in)

    rnn(t_in, (h0,), my_cell)

    t_in_time_major = Tensor(np.random.rand(3, 2, 4).astype(np.float32), TensorConfig((3, 2, 4), "float32", "cpu"))
    rnn(t_in_time_major, (h0,), my_cell, config=RNNConfig(time_major=True, return_all_outputs=False))
    rnn(t_in, (h0,), my_cell, config=RNNConfig(time_major=False, return_all_outputs=False, go_backwards=True))

    bidir_inputs = BidirectionalInputs(forward_inputs=t_in, backward_inputs=t_in, forward_initial_state=(h0,), backward_initial_state=(h0,))

    bidirectional(bidir_inputs, my_cell)
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="sum"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="mul"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="ave"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="none"))

    dev_wrapper = RNNCellDeviceWrapper(my_cell, "cpu")
    dev_wrapper(Tensor(np.random.rand(2, 4).astype(np.float32), TensorConfig((2, 4), "float32", "cpu")), (h0,))

    drop_wrapper = RNNCellDropoutWrapper(my_cell, config=DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
    drop_wrapper(Tensor(np.random.rand(2, 4).astype(np.float32), TensorConfig((2, 4), "float32", "cpu")), (h0,))

    # for residual to work, the input size must match output size of cell, which is 8
    def my_cell_res(x, state):
        return simple_rnn_cell(x, state, Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), "float32", "cpu")), rec_kernel, bias)

    res_wrapper = RNNCellResidualWrapper(my_cell_res)
    res_wrapper(Tensor(np.random.rand(2, 8).astype(np.float32), TensorConfig((2, 8), "float32", "cpu")), (h0,))

    def my_res_fn(x, y):
        return y

    res_wrapper2 = RNNCellResidualWrapper(my_cell_res, residual_fn=my_res_fn)
    res_wrapper2(Tensor(np.random.rand(2, 8).astype(np.float32), TensorConfig((2, 8), "float32", "cpu")), (h0,))

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
