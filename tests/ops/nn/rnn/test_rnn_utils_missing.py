import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.rnn_utils import DropoutWrapperConfig, RNNCellDropoutWrapper


def create_eager_tensor(data):
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


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
