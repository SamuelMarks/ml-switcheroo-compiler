"""Test module."""

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_gather, _ipc_all_reduce, _ipc_reduce_scatter


def test_numpy_distributed_ipc():
    assert _ipc_all_gather("tensor", 0, "mesh") == "tensor"
    assert _ipc_reduce_scatter("tensor", "sum", 0, "mesh") == "tensor"
    assert _ipc_all_reduce("tensor", "sum", "mesh") == "tensor"
