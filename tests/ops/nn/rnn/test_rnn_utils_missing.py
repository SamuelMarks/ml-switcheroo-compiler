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
        config = DropoutWrapperConfig(input_keep_prob=1.0, output_keep_prob=1.0)
        wrapper = RNNCellDropoutWrapper(MockCell(), config)

        inputs = create_eager_tensor(np.ones((2, 2)))
        state = (create_eager_tensor(np.ones((2, 2))),)

        out, new_state = wrapper(inputs, state)
        assert out is inputs
        assert new_state is state
