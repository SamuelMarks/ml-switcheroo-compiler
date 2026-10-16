"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _ipc_all_gather, _ipc_all_reduce, _ipc_reduce_scatter

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_numpy_distributed_ipc() -> None:
    """Test function test_numpy_distributed_ipc."""
    assert cast("Callable[..., EagerTensorProtocol]", _ipc_all_gather)("tensor", 0, "mesh") == "tensor"
    assert cast("Callable[..., EagerTensorProtocol]", _ipc_reduce_scatter)("tensor", "sum", 0, "mesh") == "tensor"
    assert cast("Callable[..., EagerTensorProtocol]", _ipc_all_reduce)("tensor", "sum", "mesh") == "tensor"
