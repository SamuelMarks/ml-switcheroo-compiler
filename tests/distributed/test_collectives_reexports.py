"""Tests for test_collectives_reexports."""

from __future__ import annotations

import ml_switcheroo_compiler.distributed.collectives as dist_collectives


class _MockBackendModule:
    """Mock backend module implementing zeros, array, and asarray."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Output shape.

        Returns:
            tuple[str, tuple[int, ...]]: Mock zeros descriptor.
        """
        return ("zeros", shape)

    def array(self, data: object, dtype: object = None) -> tuple[str, object, object]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object): Optional data type.

        Returns:
            tuple[str, object, object]: Mock array descriptor.
        """
        return ("array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Mock asarray descriptor.
        """
        return ("asarray", data)


class _MockItemObject:
    """Mock object implementing item()."""

    def item(self) -> float:
        """Return scalar item value.

        Returns:
            float: Scalar float value.
        """
        return 42.5


class _MockDaskDtypeWrapper:
    """Mock dtype object with value attribute."""

    def __init__(self, val: str) -> None:
        """Initialize mock dtype wrapper.

        Args:
            val (str): Type name.
        """
        self.value = val


def _sample_add_fn(x: int, y: int = 10) -> int:
    """Add two integers.

    Args:
        x (int): First int.
        y (int): Second int.

    Returns:
        int: Sum of x and y.
    """
    return x + y


def _sample_shape_fn(a: int, b: int) -> tuple[int, int]:
    """Sample shape function returning tuple.

    Args:
        a (int): First dimension.
        b (int): Second dimension.

    Returns:
        tuple[int, int]: Pair (a, b).
    """
    return (a, b)


def test_distributed_collectives_reexports() -> None:
    """Verify that all re-exported collective symbols in distributed.collectives exist."""
    expected_symbols = [
        "DEFAULT_CHUNK_SIZE",
        "HIGH_WATERMARK_BYTES",
        "LOW_WATERMARK_BYTES",
        "DistributedBarrier",
        "WebRTCDataChannelStream",
        "_BACKEND_COLLECTIVE_REGISTRY",
        "_recv_data",
        "_send_data",
        "all_gather",
        "all_reduce",
        "all_to_all",
        "broadcast",
        "dispatch_collective",
        "get_collective_backend",
        "recursive_halving_doubling_all_gather",
        "reduce_scatter",
        "register_collective_backend",
        "ring_all_reduce",
        "ring_reduce_scatter",
    ]
    for sym in expected_symbols:
        assert hasattr(dist_collectives, sym), f"Missing symbol {sym}"
    assert dist_collectives.__all__ == expected_symbols
