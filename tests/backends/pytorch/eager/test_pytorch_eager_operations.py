"""Tests for test_pytorch_eager_operations."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.pytorch.eager import (
    _execute_accumulate_n,
    _execute_broadcast_to,
    _execute_cast,
    _execute_ragged_tensor_to_dense,
    _execute_tensor_scatter_add,
    _execute_tensor_scatter_max,
    _execute_tensor_scatter_min,
    _execute_tensor_scatter_update,
    _get_torch_reduce_op,
    _torch_all_gather,
    _torch_all_reduce,
    _torch_all_to_all,
    _torch_reduce_scatter,
    _torch_tensordot,
    _torch_variance,
)
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from ml_switcheroo_compiler.backends.eager_registry import EagerValue

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")


def test_accumulate_n() -> None:
    """Test function test_accumulate_n."""
    res = _execute_accumulate_n([torch.tensor(1), torch.tensor(2)])
    assert res == 3
    with pytest.raises(ValueError):
        _execute_accumulate_n([])


def test_scatter_ops() -> None:
    """Test function test_scatter_ops."""
    tensor = torch.zeros((3, 3))
    indices = torch.tensor([[0, 0], [1, 1]])
    updates = torch.tensor([1.0, 2.0])

    res1 = _execute_tensor_scatter_max(tensor, indices, updates)
    assert (
        as_tensor(
            getattr(res1, "__getitem__")(
                (
                    0,
                    0,
                )
            )
        )
        == 1.0
    )

    res2 = _execute_tensor_scatter_min(tensor, indices, updates)
    assert (
        as_tensor(
            getattr(res2, "__getitem__")(
                (
                    0,
                    0,
                )
            )
        )
        == 0.0
    )

    res3 = _execute_tensor_scatter_update(tensor, indices, updates)
    assert (
        as_tensor(
            getattr(res3, "__getitem__")(
                (
                    0,
                    0,
                )
            )
        )
        == 1.0
    )

    res4 = _execute_tensor_scatter_add(tensor, indices, updates)
    assert (
        as_tensor(
            getattr(res4, "__getitem__")(
                (
                    0,
                    0,
                )
            )
        )
        == 1.0
    )


def test_broadcast_to() -> None:
    """Test function test_broadcast_to."""
    t = torch.tensor(1.0)
    res = _execute_broadcast_to(t, shape=(2, 2))
    assert as_tensor(getattr(res, "shape")) == (2, 2)


def test_cast() -> None:
    """Test function test_cast."""
    t = torch.tensor(1.0)

    class FakeDtype:
        """Test class FakeDtype."""

        value = "float8_e4m3fn"

    as_callable(_execute_cast)(t, dtype=as_callable(FakeDtype)())

    class FakeDtype2:
        """Test class FakeDtype2."""

        value = "float8_e5m2"

    as_callable(_execute_cast)(t, dtype=as_callable(FakeDtype2)())

    class FakeDtype3:
        """Test class FakeDtype3."""

        value = "int4"

    as_callable(_execute_cast)(t, dtype=as_callable(FakeDtype3)())

    class FakeDtype4:
        """Test class FakeDtype4."""

        value = "bfloat16"

    as_callable(_execute_cast)(t, dtype=as_callable(FakeDtype4)())

    class FakeDtype5:
        """Test class FakeDtype5."""

        value = "float16"

    as_callable(_execute_cast)(t, dtype=as_callable(FakeDtype5)())

    class FakeDtype6:
        """Test class FakeDtype6."""

        value = "int32"

    res_cast = as_callable(_execute_cast)(t, dtype=as_callable(FakeDtype6)())
    assert getattr(res_cast, "dtype") == torch.int32


def test_ragged_tensor_to_dense() -> None:
    # List of tensors
    """Test function test_ragged_tensor_to_dense."""
    rt = [torch.tensor([1.0, 2.0]), torch.tensor([1.0])]
    res = _execute_ragged_tensor_to_dense(rt)
    assert as_tensor(getattr(res, "shape")) == (2, 2)

    # Dict form
    rt2 = {"values": torch.tensor([1.0, 2.0, 3.0]), "row_splits": torch.tensor([0, 2, 3])}
    res2 = _execute_ragged_tensor_to_dense(rt2)
    assert as_tensor(getattr(res2, "shape")) == (2, 2)

    # Object form
    class MockRT:
        """Test class MockRT."""

        values = torch.tensor([1.0, 2.0, 3.0])
        row_splits = torch.tensor([0, 2, 3])

    res3 = as_callable(_execute_ragged_tensor_to_dense)(as_callable(MockRT)())
    assert as_tensor(getattr(res3, "shape")) == (2, 2)

    # Empty
    assert _execute_ragged_tensor_to_dense([]) == []

    # Nested Tensor
    class NestedRT:
        """Test class NestedRT."""

        is_nested = True

        def to_padded_tensor(self, padding: EagerValue) -> EagerValue:
            """Test function to_padded_tensor.

            Args:
                padding: Test parameter `padding`.
            """
            return torch.tensor([[1.0, padding], [padding, padding]])

    assert as_tensor(getattr(as_callable(_execute_ragged_tensor_to_dense)(as_callable(NestedRT)(), default_value=0.0), "shape")) == (2, 2)


def test_torch_utilities() -> None:
    """Test function test_torch_utilities."""
    t = torch.tensor([1.0, 2.0])
    res = _torch_variance(t, ddof=1)
    assert as_tensor(getattr(res, "item"))() == 0.5

    res = _torch_tensordot(t, t, axes=1)
    assert as_tensor(getattr(res, "item"))() == 5.0


def test_power_iteration() -> None:
    """Test function test_power_iteration."""
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_power_iteration

    w = torch.eye(2)
    v, u, s = as_callable(_execute_power_iteration)(w, num_iters=2)
    assert as_tensor(getattr(s, "item"))() > 0


def test_one_hot() -> None:
    """Test function test_one_hot."""
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_one_hot

    indices = torch.tensor([0, 1, 2])
    res = _execute_one_hot(indices, 3)
    assert as_tensor(getattr(res, "shape")) == (3, 3)
    res2 = _execute_one_hot(indices, 3, on_value=2.0, off_value=1.0, axis=0)
    assert as_tensor(getattr(res2, "shape")) == (3, 3)


def test_cum_ops() -> None:
    """Test function test_cum_ops."""
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_cumlogsumexp, _execute_cummax, _execute_cummin

    t = torch.tensor([1.0, 2.0, 3.0])
    assert as_tensor(getattr(_execute_cummax(t, dim=0), "shape")) == (3,)
    assert as_tensor(getattr(_execute_cummin(t, dim=0), "shape")) == (3,)
    assert as_tensor(getattr(_execute_cumlogsumexp(t, dim=0), "shape")) == (3,)


def test_execute_op_various_branches() -> None:
    """Test function test_execute_op_various_branches."""
    from ml_switcheroo_compiler.backends.pytorch.eager import execute_op

    # Branch 474->476 where schema has op but resolve_target_api returns None
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=None):
        res_atan = as_callable(execute_op)(None, "Atan2", as_callable(torch.tensor)(1.0), as_callable(torch.tensor)(1.0))
        assert as_tensor(getattr(res_atan, "item"))() == pytest.approx(0.785398)

    # Normal mapping fallback (if any) or standard pytorch op
    res = as_callable(execute_op)(None, "Add", as_callable(torch.tensor)(1), as_callable(torch.tensor)(2))
    assert as_tensor(getattr(res, "item"))() == 3

    # Hit standard torch op via lower() directly (not in schema)
    res_direct = as_callable(execute_op)(None, "ceil", as_callable(torch.tensor)(1.5))
    assert as_tensor(getattr(res_direct, "item"))() == 2.0

    # Custom op map hit
    res_onehot = as_callable(execute_op)(None, "OneHot", as_callable(torch.tensor)([0]), depth=2)
    assert as_tensor(getattr(res_onehot, "shape")) == (1, 2)

    # Capitalized string translation to F. or linalg
    res2 = as_callable(execute_op)(None, "Pad", as_callable(torch.tensor)([1.0]), pad=(1, 1))
    assert as_tensor(getattr(res2, "shape")) == (3,)

    # Global registry fallback
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    global_eager_registry.register("MyGlobalFakeOp")(lambda b, x: x * 10)
    res3 = as_callable(execute_op)(None, "MyGlobalFakeOp", as_callable(torch.tensor)(5))
    assert as_tensor(getattr(res3, "item"))() == 50

    with pytest.raises(Exception, match=r".*"):
        as_callable(execute_op)(None, "UnknownFakeOpThatWillNeverExist", as_callable(torch.tensor)(1))


def test_distributed_ops() -> None:
    """Test function test_distributed_ops."""
    t = torch.tensor([1.0, 2.0])
    assert _get_torch_reduce_op("sum") == dist.ReduceOp.SUM
    assert _get_torch_reduce_op("prod") == dist.ReduceOp.PRODUCT
    assert _get_torch_reduce_op("min") == dist.ReduceOp.MIN
    assert _get_torch_reduce_op("max") == dist.ReduceOp.MAX
    assert _get_torch_reduce_op("avg") == dist.ReduceOp.AVG
    assert _get_torch_reduce_op("unknown") == dist.ReduceOp.SUM
    with patch("torch.distributed.is_initialized", return_value=False):
        assert _torch_all_gather(t).shape == (1, 2)
        assert _torch_all_reduce(t).shape == (2,)
        assert _torch_reduce_scatter(t).shape == (2,)
        assert _torch_all_to_all(t).shape == (2,)
    with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.get_world_size", return_value=1):
        with patch("torch.distributed.all_gather"):
            _torch_all_gather(t)
        with patch("torch.distributed.all_reduce"):
            _torch_all_reduce(t)
        with patch("torch.distributed.reduce_scatter"):
            _torch_reduce_scatter(t)
        with patch("torch.distributed.all_to_all"):
            _torch_all_to_all(t)
