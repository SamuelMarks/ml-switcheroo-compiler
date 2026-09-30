"""Tests for test_dask_types."""

from __future__ import annotations

import importlib
import sys

import pytest

import ml_switcheroo_compiler.backends.dask.types as dask_types


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


def test_backends_dask_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dask backend types functions covering both branches of dtype and ImportError."""

    class _DummyDaskBackend:
        """Dummy backend class for dask types."""

        pass

    z = dask_types.zeros(_DummyDaskBackend, (2, 2))
    assert z.shape == (2, 2)

    arr1 = dask_types.array(_DummyDaskBackend, [1, 2], dtype=None)
    assert arr1.compute().tolist() == [1, 2]

    arr2 = dask_types.array(_DummyDaskBackend, [3, 4], dtype="float32")
    assert arr2.compute().tolist() == [3.0, 4.0]

    arr3 = dask_types.array(_DummyDaskBackend, [5, 6], dtype=_MockDaskDtypeWrapper("int32"))
    assert arr3.compute().tolist() == [5, 6]

    as_arr = dask_types.asarray(_DummyDaskBackend, [7, 8])
    assert as_arr.compute().tolist() == [7, 8]

    item_val = dask_types.item(_DummyDaskBackend, as_arr[0])
    assert item_val == 7.0

    # ImportError coverage for dask_types
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)
    importlib.reload(dask_types)
    assert dask_types.da is None

    # Restore dask_types
    monkeypatch.undo()
    importlib.reload(dask_types)
    assert dask_types.da is not None
