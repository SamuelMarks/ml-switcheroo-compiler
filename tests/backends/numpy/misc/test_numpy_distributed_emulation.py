"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_gather, _ipc_all_reduce, _ipc_reduce_scatter

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_numpy_distributed_ipc() -> None:
    """Test function test_numpy_distributed_ipc."""
    assert typing.cast("EagerCallableProtocol", _ipc_all_gather)("tensor", 0, "mesh") == "tensor"
    assert typing.cast("EagerCallableProtocol", _ipc_reduce_scatter)("tensor", "sum", 0, "mesh") == "tensor"
    assert typing.cast("EagerCallableProtocol", _ipc_all_reduce)("tensor", "sum", "mesh") == "tensor"
