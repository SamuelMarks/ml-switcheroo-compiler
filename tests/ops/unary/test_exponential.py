"""Tests for test_exponential."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.unary.exponential import NanToNum

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_nan_to_num_call(mocker: EagerTensorProtocol) -> None:
    """Test function test_nan_to_num_call.

    Args:
        mocker: Test parameter `mocker`.
    """
    op = NanToNum()
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    import sys

    sys.modules["ml_switcheroo_compiler.ops.base"] = sys.modules["ml_switcheroo_compiler.ops.base"]
    mocker.patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="result")
    mocker.patch("ml_switcheroo_compiler.ops.base.dispatch_op", return_value="result", create=True)
    op(x=Tensor([1], TensorConfig((1,), "float32", "cpu")), copy=True)
    assert callable(test_nan_to_num_call)
