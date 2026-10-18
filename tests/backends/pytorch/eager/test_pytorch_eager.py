# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for test_pytorch_eager."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch

import ml_switcheroo_compiler.backends.pytorch.eager as pytorch_eager
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.pytorch.eager import _execute_cumlogsumexp, _execute_cummax, _execute_cummin, _execute_power_iteration, execute_op
from ml_switcheroo_compiler.backends.pytorch.types import array, asarray, item, zeros
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)
except RuntimeError as e:
    pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)
if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_pytorch_eager_coverage() -> None:
    """Test function test_pytorch_eager_coverage."""
    t = torch.tensor([1, 2, 3])
    assert cast("Callable[..., EagerTensorProtocol]", torch.equal)(cast("Callable[..., EagerTensorProtocol]", _execute_cummax)(t, dim=0), cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1, 2, 3]))
    assert cast("Callable[..., EagerTensorProtocol]", torch.equal)(cast("Callable[..., EagerTensorProtocol]", _execute_cummin)(t, dim=0), cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1, 1, 1]))
    t_float = torch.tensor([1.0, 2.0, 3.0])
    res = _execute_cumlogsumexp(t_float, dim=0)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (3,)
    assert cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "Add", cast("Callable[..., EagerTensorProtocol]", torch.tensor)(1), cast("Callable[..., EagerTensorProtocol]", torch.tensor)(2)) == 3
    assert cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "Amax", cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1, 2])) == 2
    with pytest.raises((BackendNotSupportedError, NotImplementedError, ValueError, TypeError)):
        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "UnknownFakeOp", cast("Callable[..., EagerTensorProtocol]", torch.tensor)(1))
    assert cast("Callable[..., EagerTensorProtocol]", zeros)(None, (2,)) is not None
    assert cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2]) is not None
    assert cast("Callable[..., EagerTensorProtocol]", asarray)(None, [3, 4]) is not None
    assert cast("Callable[..., EagerTensorProtocol]", item)(None, cast("Callable[..., EagerTensorProtocol]", torch.tensor)([5])) == 5


def test_execute_power_iteration() -> None:
    """Test coverage."""
    mock_w = MagicMock()
    mock_w.shape = (2, 2)
    with pytest.raises(TypeError):
        _execute_power_iteration(mock_w)
    mock_u = MagicMock()
    mock_u.shape = (2, 1)
    with pytest.raises(TypeError):
        _execute_power_iteration(mock_w, u=mock_u)


def test_execute_one_hot() -> None:
    """Test function test_execute_one_hot."""
    import torch

    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_one_hot

    indices = torch.tensor([0, 1, 2])
    res = _execute_one_hot(indices, 3)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 3)
    res2 = _execute_one_hot(indices, 3, on_value=2.0, off_value=1.0, axis=0)
    assert cast("EagerTensorProtocol", getattr(res2, "shape")) == (3, 3)


def test_execute_ragged_tensor_to_dense() -> None:
    """Test function test_execute_ragged_tensor_to_dense."""
    import torch

    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense

    rt = [torch.tensor([1, 2]), torch.tensor([1])]
    res = _execute_ragged_tensor_to_dense(rt)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_pytorch_eager_global_registry() -> None:
    """Test global registry fallback in execute_op."""
    global_eager_registry.register("MyFakeGlobalOp")(lambda backend, *args, **kwargs: "global_hit")
    assert cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "MyFakeGlobalOp") == "global_hit"


def test_pytorch_eager_snake_case_fallback() -> None:
    """Test snake_case fallback for F/linalg/fft/torch modules."""
    t = torch.tensor([1.0, 2.0])
    res = cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "LogSoftmax", t, dim=0)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)


def test_pytorch_eager_not_supported() -> None:
    """Test BackendNotSupportedError."""
    with pytest.raises(BackendNotSupportedError, match="Operation 'TotallyFakeOp' is not implemented"):
        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "TotallyFakeOp")


def test_pytorch_eager_fallback_exception() -> None:
    """Test exception during execution in fallback logic."""
    with pytest.raises(TypeError):
        cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "LogSoftmax", "not_a_tensor", dim=0)


def test_pytorch_eager_coverage_2() -> None:
    """Test function test_pytorch_eager_coverage_2."""
    res = pytorch_eager._execute_accumulate_n([torch.tensor(1), torch.tensor(2)])
    assert cast("Callable[..., EagerTensorProtocol]", int)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "item")))() if cast("Callable[..., EagerTensorProtocol]", hasattr)(res, "item") else res) == 3
    res = pytorch_eager._execute_accumulate_n(inputs=[torch.tensor(1), torch.tensor(2)])
    assert cast("Callable[..., EagerTensorProtocol]", int)(cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(res, "item")))() if cast("Callable[..., EagerTensorProtocol]", hasattr)(res, "item") else res) == 3
    with pytest.raises(ValueError):
        pytorch_eager._execute_accumulate_n(inputs=[])
    tensor = torch.zeros(2, 2)
    indices = torch.tensor([[0, 0]])
    updates = torch.tensor([1.0])
    res = pytorch_eager._execute_tensor_scatter_max(tensor, indices, updates)
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")((0, 0))) == 1.0
    res = pytorch_eager._execute_tensor_scatter_min(tensor, indices, updates)
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")((0, 0))) == 0.0
    res = pytorch_eager._execute_tensor_scatter_update(tensor, indices, updates)
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")((0, 0))) == 1.0
    res = pytorch_eager._execute_tensor_scatter_add(tensor, indices, updates)
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")((0, 0))) == 1.0
    w = torch.eye(2).unsqueeze(0)
    res = pytorch_eager._execute_power_iteration(w)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 3
    res = pytorch_eager._execute_broadcast_to(torch.tensor([1.0]), shape=(2,))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)
    res = pytorch_eager._execute_cummax(torch.tensor([1, 2]), dim=0)
    assert res is not None
    res = pytorch_eager._execute_cummin(torch.tensor([1, 2]), dim=0)
    assert res is not None
    res = pytorch_eager._execute_cumlogsumexp(torch.tensor([1.0, 2.0]), dim=0)
    assert res is not None
    res = pytorch_eager._execute_ragged_tensor_to_dense(1)
    assert res == 1
    res = pytorch_eager._torch_variance(torch.tensor([1.0, 2.0]))
    assert res is not None
    res = pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), dims=1)
    assert res is not None
    res = pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), axes=1)
    assert res is not None
    res = cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "Add", cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]), cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]))
    res = cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "Cumlogsumexp", cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0, 2.0]), dim=0)
    assert res is not None
    with pytest.raises((BackendNotSupportedError, NotImplementedError, ValueError, TypeError)):
        cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "UnknownOp")
