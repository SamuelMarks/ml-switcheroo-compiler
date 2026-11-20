"""Tests for test_frontend."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.ops.nn.conv as conv1d
import ml_switcheroo_compiler.ops.nn.conv as conv2d
import ml_switcheroo_compiler.ops.nn.conv as conv3d
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.conv_utils import _build_conv_config

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_conv_frontend_cov() -> None:
    """Test function test_conv_frontend_cov."""
    orig_eager = config.eager_mode
    orig_backend = config.backend

    class MockBackendConv(BaseGenerator):
        """Test class MockBackendConv."""

        @classmethod
        def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op_type (str): Test parameter `op_type`.
                *args (BackendArg): Test parameter `args`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            del op_type, args, kwargs
            return np.ones((2, 2, 2))

        @classmethod
        def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
            """Test function array.

            Args:
                data (BackendArg): Test parameter `data`.
                dtype (BackendArg | None): Test parameter `dtype`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            del dtype
            return data

    BackendRegistry.register("mock_conv", MockBackendConv)

    try:
        config.eager_mode = True
        config.backend = "mock_conv"

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.shape = (2, 2, 2)
                self.dtype = DType.Float32
                self.device = None
                self.data = np.ones((2, 2, 2))

        t = Tensor(
            DummyTensor(),
            TensorConfig(
                shape=(2, 2, 2),
                dtype=DType.Float32,
                device=None,
                requires_grad=False,
                trainable=False,
            ),
        )

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
    assert config.backend == orig_backend


def test_conv_frontend_cov_config_obj_not_none() -> None:
    """Test function test_conv_frontend_cov_config_obj_not_none."""
    orig_eager = config.eager_mode
    orig_backend = config.backend

    try:

        class MockBackendConv(BaseGenerator):
            """Test class MockBackendConv."""

            @classmethod
            def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function execute_op.

                Args:
                    op_type (str): Test parameter `op_type`.
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `execute_op`.
                """
                del op_type, args, kwargs
                return None

            @classmethod
            def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
                """Test function array.

                Args:
                    data (BackendArg): Test parameter `data`.
                    dtype (BackendArg | None): Test parameter `dtype`.

                Returns:
                    BackendArg: Computed output value from `array`.
                """
                del data, dtype
                return np.ones((1, 1))

        BackendRegistry.register("mock_conv", MockBackendConv)

        config.eager_mode = True
        config.backend = "mock_conv"

        class DummyTensor:
            """Test class DummyTensor."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.shape = (2, 2, 2)
                self.dtype = DType.Float32
                self.device = None
                self.data = np.ones((2, 2, 2))

        t = Tensor(
            DummyTensor(),
            TensorConfig(
                shape=(2, 2, 2),
                dtype=DType.Float32,
                device=None,
                requires_grad=False,
                trainable=False,
            ),
        )

        # conv1d
        conv1d.conv1d(
            t,
            t,
            config_obj=_build_conv_config({}, ((0, 2, 1), (2, 1, 0), (0, 2, 1))),
        )

        # conv2d
        conv2d.conv2d(
            t,
            t,
            config_obj=_build_conv_config({}, ((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))),
        )

        # conv3d
        conv3d.conv3d(
            t,
            t,
            config_obj=_build_conv_config({}, ((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3))),
        )

    finally:
        config.eager_mode = orig_eager
        config.backend = orig_backend
    assert config.backend == orig_backend
