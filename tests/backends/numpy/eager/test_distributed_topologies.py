"""Tests for test_distributed_topologies."""

from __future__ import annotations

import types
import typing

import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext
from tests.testing_utils import (
    EagerBoundCallable,
    EagerTensorProtocol,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_numpy_distributed_extra() -> None:
    """Test function test_numpy_distributed_extra."""
    from unittest.mock import MagicMock

    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dist_mod
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import (
        TCPDistributedContext,
        _np_all_gather,
        _np_all_reduce,
        _np_all_to_all,
        _np_broadcast,
        _np_reduce,
        _np_reduce_scatter,
    )

    # Test all functions without global context
    dist_mod._tcp_dist_ctx.world_size = 1
    assert np.array_equal(ensure_ndarray(_np_all_reduce(np, np.array([1, 2]))), np.array([1, 2]))
    assert np.array_equal(ensure_ndarray(_np_all_gather(np, np.array([1, 2]))), np.array([[1, 2]]))
    assert np.array_equal(ensure_ndarray(_np_broadcast(np, np.array([1, 2]))), np.array([1, 2]))
    assert np.array_equal(ensure_ndarray(_np_reduce_scatter(np, np.array([1, 2]))), np.array([1, 2]))
    assert np.array_equal(ensure_ndarray(_np_reduce(np, np.array([1, 2]))), np.array([1, 2]))
    assert np.array_equal(ensure_ndarray(_np_all_to_all(np, np.array([1, 2]))), np.array([1, 2]))

    # Mock context
    mock_ctx = MagicMock(spec=TCPDistributedContext)
    mock_ctx.world_size = 2
    mock_ctx.rank = 0

    mock_ctx.all_reduce_ring.return_value = np.array([4, 4])
    dist_mod._tcp_dist_ctx = mock_ctx

    # Test all_reduce with context for other op types
    mock_ctx.all_reduce_ring.return_value = np.array([4, 4])
    assert np.array_equal(ensure_ndarray(_np_all_reduce(np, np.array([1, 2]))), np.array([4, 4]))

    # Test all_gather with context
    mock_ctx.all_gather_tensors.return_value = [np.array([1]), np.array([2])]
    assert np.array_equal(ensure_ndarray(_np_all_gather(np, np.array([1]))), np.array([1, 2]))

    # Test broadcast with context
    assert np.array_equal(ensure_ndarray(_np_broadcast(np, np.array([1, 2]))), np.array([1]))

    # Test reduce_scatter with context for various op_types
    for op in ["sum", "prod", "max", "min", "unknown"]:
        res = _np_reduce_scatter(np, np.array([1, 2]), op, 0)
        assert extract_test_shape(res) == (1,)

    # Test reduce with context for various op_types
    for op, expected in zip(["sum", "prod", "max", "min", "unknown"], [3, 2, 2, 1, 3]):
        assert np.array_equal(
            ensure_ndarray(_np_reduce(np, np.array([1, 2]), op_type=op)),
            np.array([expected]),
        )

    # Test all_to_all with context
    res_all_to_all = _np_all_to_all(np, np.array([1, 2]))
    assert len(coerce_operand_list(res_all_to_all)) == 2
    assert np.array_equal(ensure_ndarray(coerce_operand_list(res_all_to_all)[0]), np.array([1]))

    # Clean up
    pytest.MonkeyPatch().setattr(dist_mod, "_GLOBAL_NP_DIST_CONTEXT", None, raising=False)


class _TestNumpyDistributedContextMethodsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_numpy_distributed_context_methods` split helpers."""

    MagicMock: type
    TCPDistributedContext: type
    ctx: EagerTensorProtocol
    ctx_other: EagerTensorProtocol
    ctx_tree: TCPDistributedContext
    np: types.ModuleType
    patch: EagerBoundCallable


def _test_numpy_distributed_context_methods_part_1(_state: _TestNumpyDistributedContextMethodsState) -> None:
    """Execute part 1 of test_numpy_distributed_context_methods.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import MagicMock, mock_open, patch

    import numpy as np
    import yaml

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext

    # Test load yaml config
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"topologies": {"ring": {"key": "val"}}}))):
            ctx_yaml = TCPDistributedContext()
            assert ctx_yaml.config == {"key": "val"}

    # Test initialization with tree topology
    ctx_tree = TCPDistributedContext(world_size=2, rank=1, topology="tree")
    _state.update({"MagicMock": MagicMock, "TCPDistributedContext": TCPDistributedContext, "ctx_tree": ctx_tree})
    _state.update({"np": np, "patch": require_callable(patch)})


def _test_numpy_distributed_context_methods_part_2(_state: _TestNumpyDistributedContextMethodsState) -> None:
    """Execute part 2 of test_numpy_distributed_context_methods.

    Args:
        _state: Shared test state dictionary.
    """
    TCPDistributedContext, ctx_tree, patch = _state["TCPDistributedContext"], _state["ctx_tree"], _state["patch"]
    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener"):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
            pytest.MonkeyPatch().setattr(mock_client, "side_effect", ConnectionRefusedError, raising=False)
            with patch("time.sleep"):
                with patch("threading.Thread"):
                    ctx_tree.initialize()  # Will break after retries

    ctx_other = TCPDistributedContext(world_size=2, rank=1, topology="other")
    _state.update({"ctx_other": ctx_other})
    _state.update(
        {"TCPDistributedContext": TCPDistributedContext, "ctx_tree": ctx_tree, "patch": require_callable(patch)}
    )


def _test_numpy_distributed_context_methods_part_3(_state: _TestNumpyDistributedContextMethodsState) -> None:
    """Execute part 3 of test_numpy_distributed_context_methods.

    Args:
        _state: Shared test state dictionary.
    """
    TCPDistributedContext, ctx_other, patch = _state["TCPDistributedContext"], _state["ctx_other"], _state["patch"]
    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener"):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client"):
            with patch("threading.Thread"):
                ctx_other.initialize()

    ctx = TCPDistributedContext(world_size=2, rank=0)
    _state.update({"ctx": ctx})
    _state.update(
        {"TCPDistributedContext": TCPDistributedContext, "ctx_other": ctx_other, "patch": require_callable(patch)}
    )


def _test_numpy_distributed_context_methods_part_4(_state: _TestNumpyDistributedContextMethodsState) -> None:
    """Execute part 4 of test_numpy_distributed_context_methods.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, TCPDistributedContext, ctx, patch = (
        _state["MagicMock"],
        _state["TCPDistributedContext"],
        _state["ctx"],
        _state["patch"],
    )
    with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Listener") as mock_listener:
        with patch("ml_switcheroo_compiler.backends.numpy.eager.distributed.Client") as mock_client:
            mock_listener_inst = MagicMock()
            mock_listener.return_value = mock_listener_inst

            mock_client.return_value = MagicMock()

            # Make accept return a dummy connection once
            mock_listener_inst.accept.return_value = MagicMock()

            # Use a short timeout so it doesn't hang
            with patch("threading.Thread") as mock_thread:
                ctx.initialize()
                assert mock_thread.called

            # Manually run the target
            target = mock_thread.call_args[1]["target"]
            target()

            assert len(ctx.recv_conns) == 1

    # Test all_gather_tensors
    # Let's mock a simple socket scenario
    ctx = TCPDistributedContext(world_size=2, rank=0)
    pytest.MonkeyPatch().setattr(ctx, "connections", [MagicMock()], raising=False)
    _state.update({"ctx": ctx})
    _state.update(
        {"MagicMock": MagicMock, "TCPDistributedContext": TCPDistributedContext, "patch": require_callable(patch)}
    )


def _test_numpy_distributed_context_methods_part_5(_state: _TestNumpyDistributedContextMethodsState) -> None:
    """Execute part 5 of test_numpy_distributed_context_methods.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, TCPDistributedContext, ctx, np = (
        _state["MagicMock"],
        _state["TCPDistributedContext"],
        _state["ctx"],
        _state["np"],
    )
    patch = _state["patch"]
    with patch("pickle.dumps", return_value=b"data"):
        with patch("struct.pack", return_value=b"len"):
            # Mock receive length then data
            pytest.MonkeyPatch().setattr(ctx.connections[0].recv, "side_effect", [b"len", b"data"], raising=False)
            with patch("struct.unpack", return_value=(4,)):
                with patch("pickle.loads", return_value=np.array([1])):
                    res = ctx.all_gather_tensors(np.array([1]))
                    assert len(res) == 2  # self + 1 peer

    ctx_single = TCPDistributedContext(world_size=1, rank=0)
    res = ctx_single.all_gather_tensors(np.array([1]))
    assert len(res) == 1
    res2 = ctx_single.all_reduce_ring(np.array([1]))
    assert len(res2) == 1

    # Test all_reduce_ring operations
    ctx_ring = TCPDistributedContext(world_size=2, rank=0)
    pytest.MonkeyPatch().setattr(ctx_ring, "send_conns", [MagicMock()], raising=False)
    pytest.MonkeyPatch().setattr(ctx_ring, "recv_conns", [MagicMock()], raising=False)

    for op in ["prod", "max", "min"]:
        ctx_ring.recv_conns[0].recv.return_value = np.array([2])
        res = ctx_ring.all_reduce_ring(np.array([2]), op_type=op)

    ctx_ring.all_gather_tensors(np.array([1]))


def test_numpy_distributed_context_methods() -> None:
    """Test function test_numpy_distributed_context_methods."""
    _state: _TestNumpyDistributedContextMethodsState = {}
    _test_numpy_distributed_context_methods_part_1(_state)
    _test_numpy_distributed_context_methods_part_2(_state)
    _test_numpy_distributed_context_methods_part_3(_state)
    _test_numpy_distributed_context_methods_part_4(_state)
    _test_numpy_distributed_context_methods_part_5(_state)
    assert len(_state) > 0
