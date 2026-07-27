"""Test module."""

from ml_switcheroo_compiler.backends.numpy.distributed.dummy import _dummy_all_gather, _dummy_all_reduce, _dummy_reduce_scatter


def test_numpy_distributed_dummy():
    assert _dummy_all_gather("tensor", 0, "mesh") == "tensor"
    assert _dummy_reduce_scatter("tensor", "sum", 0, "mesh") == "tensor"
    assert _dummy_all_reduce("tensor", "sum", "mesh") == "tensor"
