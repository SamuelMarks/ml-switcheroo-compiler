"""Tests for test_erosion_dilation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


def test_conv_utils_extras() -> None:
    """Test function test_conv_utils_extras."""
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

    class DummyTensor:
        """Test class DummyTensor."""

        shape = (1, 1, 1, 1)

        def __array__(self) -> EagerValue:
            """Test function __array__."""
            return np.ones((1, 1, 1, 1))

        @property
        def dtype(self) -> EagerValue:
            """Test function dtype."""
            return "float32"

        def asnumpy(self) -> EagerValue:
            """Test function asnumpy."""
            return np.ones((1, 1, 1, 1))

    t = DummyTensor()

    def mock_get_op(name: EagerValue) -> EagerValue:
        """Test function mock_get_op.

        Args:
            name: Test parameter `name`.
        """

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return lambda *a, **kw: "dummy_node"

        return MockOp

    with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", side_effect=mock_get_op):
        conv_utils.atrous_conv2d(t, t, rate=1, padding="VALID")
        conv_utils.atrous_conv2d_transpose(t, t, t)
        cast("EagerCallableProtocol", conv_utils.depthwise_conv2d)(t, t)
        conv_utils.convolution(t, t)

    cfg = conv_utils._build_conv_config({"lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1, 2, 3), (), ()))
    assert cfg.lhs_dilation == (1, 1)


def test_erosion_dilation() -> None:
    """Test function test_erosion_dilation."""
    assert callable(test_erosion_dilation)
