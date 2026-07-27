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
