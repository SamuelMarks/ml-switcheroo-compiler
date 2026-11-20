"""Tests for test_erosion_dilation."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

from tests.testing_utils import MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_conv_utils_extras() -> None:
    """Test function test_conv_utils_extras."""
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.shape = (1, 1, 1, 1)

        def __array__(self) -> BackendArg:
            """Test function __array__.

            Returns:
                BackendArg: Computed output value from `__array__`.
            """
            return np.ones((1, 1, 1, 1))

        @property
        def dtype(self) -> BackendArg:
            """Test function dtype.

            Returns:
                BackendArg: Computed output value from `dtype`.
            """
            return "float32"

        def asnumpy(self) -> BackendArg:
            """Test function asnumpy.

            Returns:
                BackendArg: Computed output value from `asnumpy`.
            """
            return np.ones((1, 1, 1, 1))

    t = DummyTensor()

    def mock_get_op(name: BackendArg) -> BackendArg:
        """Test function mock_get_op.

        Args:
            name (BackendArg): Test parameter `name`.

        Returns:
            BackendArg: Computed output value from `mock_get_op`.
        """

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return lambda *a, **kw: "dummy_node"

        return MockOp

    with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", side_effect=mock_get_op):
        conv_utils.atrous_conv2d(t, t, rate=1, padding="VALID")
        conv_utils.atrous_conv2d_transpose(t, t, t)
        require_callable(conv_utils.depthwise_conv2d)(t, t)
        conv_utils.convolution(t, t)

    cfg = conv_utils._build_conv_config({"lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1, 2, 3), (), ()))
    assert cfg.lhs_dilation == (1, 1)
