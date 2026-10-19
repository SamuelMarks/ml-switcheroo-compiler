"""Tests for test_frontend."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast

import numpy as np

import ml_switcheroo_compiler.ops.nn.conv1d as conv1d
import ml_switcheroo_compiler.ops.nn.conv2d as conv2d
import ml_switcheroo_compiler.ops.nn.conv3d as conv3d
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_conv_frontend_cov() -> None:
    """Test function test_conv_frontend_cov."""
    orig_eager = config.eager_mode
    orig_backend = config.backend

    class MockBackendConv:
        """Test class MockBackendConv."""

        @classmethod
        def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op_name: Test parameter `op_name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return np.ones((2, 2, 2))

        @classmethod
        def array(cls, data: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                data: Test parameter `data`.
            """
            return data

    cast("EagerCallableProtocol", BackendRegistry.register)("mock_conv", MockBackendConv)

    try:
        config.eager_mode = True
        config.backend = "mock_conv"

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.shape = (2, 2, 2)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None
                self.data = np.ones((2, 2, 2))

        t = cast("EagerCallableProtocol", Tensor)(cast("EagerCallableProtocol", DummyTensor)(), cast("EagerCallableProtocol", types.SimpleNamespace)(shape=(2, 2, 2), dtype=cast("EagerCallableProtocol", DummyTensor)().dtype, device=None, requires_grad=False, trainable=False))

        # conv1d
        cast("EagerCallableProtocol", conv1d.conv1d)(t, t, config_obj=None)
        cast("EagerCallableProtocol", conv1d.conv1d_transpose)(t, t)
        cast("EagerCallableProtocol", conv1d.depthwise_conv1d)(t, t, config_obj=None)
        cast("EagerCallableProtocol", conv1d.separable_conv1d)(t, t, t)

        # conv2d
        cast("EagerCallableProtocol", conv2d.conv2d)(t, t, config_obj=None)
        cast("EagerCallableProtocol", conv2d.conv2d_transpose)(t, t)
        cast("EagerCallableProtocol", conv2d.depthwise_conv2d)(t, t, config_obj=None)
        cast("EagerCallableProtocol", conv2d.separable_conv2d)(t, t, t)

        # conv3d
        cast("EagerCallableProtocol", conv3d.conv3d)(t, t, config_obj=None)
        cast("EagerCallableProtocol", conv3d.conv3d_transpose)(t, t)

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend
    assert config.backend == orig_backend


def test_conv_frontend_cov_config_obj_not_none() -> None:
    """Test function test_conv_frontend_cov_config_obj_not_none."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.config import config

    orig_eager = config.eager_mode
    orig_backend = config.backend

    try:

        class MockBackendConv:
            """Test class MockBackendConv."""

            @classmethod
            def execute_op(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
                """Test function execute_op.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return None

            @classmethod
            def array(cls, x: EagerValue) -> EagerValue:
                """Test function array.

                Args:
                    x: Test parameter `x`.
                """
                return types.SimpleNamespace(shape=(1, 1))

        cast("EagerCallableProtocol", BackendRegistry.register)("mock_conv", MockBackendConv)

        config.eager_mode = True
        config.backend = "mock_conv"

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.shape = (2, 2, 2)
                self.dtype = types.SimpleNamespace(value="float32")
                self.device = None
                self.data = np.ones((2, 2, 2))

        t = cast("EagerCallableProtocol", Tensor)(cast("EagerCallableProtocol", DummyTensor)(), cast("EagerCallableProtocol", types.SimpleNamespace)(shape=(2, 2, 2), dtype=cast("EagerCallableProtocol", DummyTensor)().dtype, device=None, requires_grad=False, trainable=False))

        # conv1d
        cast("EagerCallableProtocol", conv1d.conv1d)(t, t, config_obj=cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(conv1d, "_build_conv_config")))({}, ((0, 2, 1), (2, 1, 0), (0, 2, 1))))

        # conv2d
        cast("EagerCallableProtocol", conv2d.conv2d)(t, t, config_obj=cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(conv2d, "_build_conv_config")))({}, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))))

        # conv3d
        cast("EagerCallableProtocol", conv3d.conv3d)(t, t, config_obj=cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(conv3d, "_build_conv_config")))({}, ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3))))

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend
    assert config.backend == orig_backend
