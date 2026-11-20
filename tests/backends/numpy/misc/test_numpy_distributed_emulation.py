"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_gather, _ipc_all_reduce, _ipc_reduce_scatter
from ml_switcheroo_compiler.distributed.device_mesh import DeviceMesh

if typing.TYPE_CHECKING:
    pass


def test_numpy_distributed_ipc() -> None:
    """Test function test_numpy_distributed_ipc."""
    mesh = DeviceMesh(shape=(1,), axis_names=("x",))
    assert _ipc_all_gather("tensor", 0, mesh) == "tensor"
    assert _ipc_reduce_scatter("tensor", "sum", 0, mesh) == "tensor"
    assert _ipc_all_reduce("tensor", "sum", mesh) == "tensor"
