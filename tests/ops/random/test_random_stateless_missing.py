import unittest.mock as mock

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import Generator, stateless_split


def create_eager_tensor(data):
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_random_stateless_missing():
    gen = Generator()

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            res1 = gen.normal((2,))
            res2 = gen.uniform((2,))
            assert isinstance(res1, Tensor)
            assert isinstance(res2, Tensor)

            seed_tensor = create_eager_tensor(np.array([42, 0]))
            res3 = stateless_split(seed_tensor, num=2)
            assert isinstance(res3, Tensor)
