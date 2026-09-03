from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

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


def test_accumulate_n():
    res = _execute_accumulate_n([torch.tensor(1), torch.tensor(2)])
    assert res == 3
    with pytest.raises(ValueError):
        _execute_accumulate_n([])


def test_scatter_ops():
    tensor = torch.zeros((3, 3))
    indices = torch.tensor([[0, 0], [1, 1]])
    updates = torch.tensor([1.0, 2.0])

    res1 = _execute_tensor_scatter_max(tensor, indices, updates)
    assert res1[0, 0] == 1.0

    res2 = _execute_tensor_scatter_min(tensor, indices, updates)
    assert res2[0, 0] == 0.0

    res3 = _execute_tensor_scatter_update(tensor, indices, updates)
    assert res3[0, 0] == 1.0

    res4 = _execute_tensor_scatter_add(tensor, indices, updates)
    assert res4[0, 0] == 1.0


def test_broadcast_to():
    t = torch.tensor(1.0)
    res = _execute_broadcast_to(t, shape=(2, 2))
    assert res.shape == (2, 2)


def test_cast():
    t = torch.tensor(1.0)

    class FakeDtype:
        value = "float8_e4m3fn"

    _execute_cast(t, dtype=FakeDtype())

    class FakeDtype2:
        value = "float8_e5m2"

    _execute_cast(t, dtype=FakeDtype2())

    class FakeDtype3:
        value = "int4"

    _execute_cast(t, dtype=FakeDtype3())

    class FakeDtype4:
        value = "bfloat16"

    _execute_cast(t, dtype=FakeDtype4())

    class FakeDtype5:
        value = "float16"

    _execute_cast(t, dtype=FakeDtype5())

    class FakeDtype6:
        value = "int32"

    _execute_cast(t, dtype=FakeDtype6())


def test_ragged_tensor_to_dense():
    # List of tensors
    rt = [torch.tensor([1.0, 2.0]), torch.tensor([1.0])]
    res = _execute_ragged_tensor_to_dense(rt)
    assert res.shape == (2, 2)

    # Dict form
    rt2 = {"values": torch.tensor([1.0, 2.0, 3.0]), "row_splits": torch.tensor([0, 2, 3])}
    res2 = _execute_ragged_tensor_to_dense(rt2)
    assert res2.shape == (2, 2)

    # Object form
    class MockRT:
        values = torch.tensor([1.0, 2.0, 3.0])
        row_splits = torch.tensor([0, 2, 3])

    res3 = _execute_ragged_tensor_to_dense(MockRT())
    assert res3.shape == (2, 2)

    # Empty
    assert _execute_ragged_tensor_to_dense([]) == []

    # Nested Tensor
    class NestedRT:
        is_nested = True

        def to_padded_tensor(self, padding):
            return torch.tensor([[1.0, padding], [padding, padding]])

    assert _execute_ragged_tensor_to_dense(NestedRT(), default_value=0.0).shape == (2, 2)


def test_torch_utilities():
    t = torch.tensor([1.0, 2.0])
    res = _torch_variance(t, ddof=1)
    assert res.item() == 0.5

    res = _torch_tensordot(t, t, axes=1)
    assert res.item() == 5.0


def test_power_iteration():
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_power_iteration

    w = torch.eye(2)
    v, u, s = _execute_power_iteration(w, num_iters=2)
    assert s.item() > 0


def test_one_hot():
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_one_hot

    indices = torch.tensor([0, 1, 2])
    res = _execute_one_hot(indices, 3)
    assert res.shape == (3, 3)
    res2 = _execute_one_hot(indices, 3, on_value=2.0, off_value=1.0, axis=0)
    assert res2.shape == (3, 3)


def test_cum_ops():
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_cumlogsumexp, _execute_cummax, _execute_cummin

    t = torch.tensor([1.0, 2.0, 3.0])
    assert _execute_cummax(t, dim=0).shape == (3,)
    assert _execute_cummin(t, dim=0).shape == (3,)
    assert _execute_cumlogsumexp(t, dim=0).shape == (3,)


def test_execute_op_fallback():
    from ml_switcheroo_compiler.backends.pytorch.eager import execute_op

    # Normal mapping fallback (if any) or standard pytorch op
    res = execute_op(None, "Add", torch.tensor(1), torch.tensor(2))
    assert res.item() == 3

    # Hit standard torch op via lower() directly (not in schema)
    res_direct = execute_op(None, "ceil", torch.tensor(1.5))
    assert res_direct.item() == 2.0

    # Custom op map hit
    res_onehot = execute_op(None, "OneHot", torch.tensor([0]), depth=2)
    assert res_onehot.shape == (1, 2)

    # Capitalized string translation to F. or linalg
    res2 = execute_op(None, "Pad", torch.tensor([1.0]), pad=(1, 1))
    assert res2.shape == (3,)

    # Global registry fallback
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    global_eager_registry.register("MyGlobalFakeOp")(lambda b, x: x * 10)
    res3 = execute_op(None, "MyGlobalFakeOp", torch.tensor(5))
    assert res3.item() == 50

    with pytest.raises(Exception):
        execute_op(None, "UnknownFakeOpThatWillNeverExist", torch.tensor(1))


def test_distributed_ops():
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
