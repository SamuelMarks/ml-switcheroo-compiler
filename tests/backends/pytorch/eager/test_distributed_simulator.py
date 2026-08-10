import pytest


def test_pytorch_distributed_simulators():
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        pytest.skip("PyTorch not installed")

    from ml_switcheroo_compiler.backends.pytorch.eager import _get_torch_reduce_op, _torch_all_gather, _torch_all_reduce, _torch_all_to_all, _torch_reduce_scatter

    tensor = torch.tensor([1, 2, 3])

    # Uninitialized branch
    res = _torch_all_gather(tensor, axis=0)
    assert res.shape == (1, 3)

    res = _torch_all_reduce(tensor)
    assert torch.equal(res, tensor)

    res = _torch_reduce_scatter(tensor)
    assert torch.equal(res, tensor)

    res = _torch_all_to_all(tensor)
    assert torch.equal(res, tensor)

    # Initialized branch
    from unittest import mock

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=1),
        mock.patch("torch.distributed.all_gather", lambda out, inp, **kw: out.insert(0, inp) or out.pop()),
        mock.patch("torch.distributed.all_reduce", lambda tensor, op, **kw: None),
        mock.patch("torch.distributed.reduce_scatter", lambda out, inp, op, **kw: out.copy_(inp[0])),
        mock.patch("torch.distributed.all_to_all", lambda out, inp, **kw: out.insert(0, inp[0]) or out.pop()),
    ):
        # test _get_torch_reduce_op
        assert _get_torch_reduce_op("sum") == dist.ReduceOp.SUM
        assert _get_torch_reduce_op("prod") == dist.ReduceOp.PRODUCT
        assert _get_torch_reduce_op("min") == dist.ReduceOp.MIN
        assert _get_torch_reduce_op("max") == dist.ReduceOp.MAX
        assert _get_torch_reduce_op("avg") == dist.ReduceOp.AVG
        assert _get_torch_reduce_op("mean") == dist.ReduceOp.AVG
        assert _get_torch_reduce_op("unknown") == dist.ReduceOp.SUM

        _torch_all_gather(tensor, axis=0)
        _torch_all_reduce(tensor, op_type="sum")
        _torch_reduce_scatter(tensor, op_type="sum", axis=0)
        _torch_all_to_all(tensor, split_axis=0, concat_axis=0)
