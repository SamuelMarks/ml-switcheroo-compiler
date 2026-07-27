import types

import numpy as np

import ml_switcheroo_compiler.ops.nn.conv1d as conv1d
import ml_switcheroo_compiler.ops.nn.conv2d as conv2d
import ml_switcheroo_compiler.ops.nn.conv3d as conv3d
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor


def test_conv_frontend_cov():
    orig_eager = config.eager_mode
    orig_backend = config.backend

    class MockBackendConv:
        @classmethod
        def execute_op(cls, op_name, *args, **kwargs):
            return np.ones((2, 2, 2))

        @classmethod
        def array(cls, data):
            return data

    BackendRegistry.register("mock_conv", MockBackendConv)

    try:
        config.eager_mode = True
        config.backend = "mock_conv"

        class DummyTensor:
            def __init__(self):
                self.shape = (2, 2, 2)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None
                self.data = np.ones((2, 2, 2))

        t = Tensor(DummyTensor(), types.SimpleNamespace(shape=(2, 2, 2), dtype=DummyTensor().dtype, device=None, requires_grad=False, trainable=False))

        # conv1d
        conv1d.conv1d(t, t, config_obj=None)
        conv1d.conv1d_transpose(t, t)
        conv1d.depthwise_conv1d(t, t, config_obj=None)
        conv1d.separable_conv1d(t, t, t)

        # conv2d
        conv2d.conv2d(t, t, config_obj=None)
        conv2d.conv2d_transpose(t, t)
        conv2d.depthwise_conv2d(t, t, config_obj=None)
        conv2d.separable_conv2d(t, t, t)

        # conv3d
        conv3d.conv3d(t, t, config_obj=None)
        conv3d.conv3d_transpose(t, t)

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend


def test_conv_frontend_cov_config_obj_not_none():
    from ml_switcheroo_compiler.core.config import config

    orig_eager = config.eager_mode
    orig_backend = config.backend

    try:
        config.eager_mode = True
        config.backend = "mock_conv"

        class DummyTensor:
            def __init__(self):
                self.shape = (2, 2, 2)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None
                self.data = np.ones((2, 2, 2))

        t = Tensor(DummyTensor(), types.SimpleNamespace(shape=(2, 2, 2), dtype=DummyTensor().dtype, device=None, requires_grad=False, trainable=False))

        # conv1d
        conv1d.conv1d(t, t, config_obj=conv1d._build_conv_config({}, ((0, 2, 1), (2, 1, 0), (0, 2, 1))))

        # conv2d
        conv2d.conv2d(t, t, config_obj=conv2d._build_conv_config({}, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))))

        # conv3d
        conv3d.conv3d(t, t, config_obj=conv3d._build_conv_config({}, ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3))))

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend
