"""Test PyTorch eager edge cases coverage."""

import pytest
import torch

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.pytorch.eager import execute_op
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_pytorch_eager_global_registry():
    """Test global registry fallback in execute_op."""
    global_eager_registry.register("MyFakeGlobalOp")(lambda backend, *args, **kwargs: "global_hit")

    assert execute_op(None, "MyFakeGlobalOp") == "global_hit"


def test_pytorch_eager_snake_case_fallback():
    """Test snake_case fallback for F/linalg/fft/torch modules."""
    # F.log_softmax is in torch.nn.functional
    # logsoftmax does not exist in torch, so AttributeError will be raised first
    # then it will fall back to snake_case 'log_softmax' and find it in F
    t = torch.tensor([1.0, 2.0])
    res = execute_op(None, "LogSoftmax", t, dim=0)
    assert res.shape == (2,)


def test_pytorch_eager_not_supported():
    """Test BackendNotSupportedError."""
    with pytest.raises(BackendNotSupportedError, match="Operation 'TotallyFakeOp' is not implemented"):
        execute_op(None, "TotallyFakeOp")


def test_pytorch_eager_fallback_exception():
    """Test exception during execution in fallback logic."""
    # F.log_softmax is found, but passing a string causes a TypeError which is caught
    with pytest.raises(TypeError):
        execute_op(None, "LogSoftmax", "not_a_tensor", dim=0)
