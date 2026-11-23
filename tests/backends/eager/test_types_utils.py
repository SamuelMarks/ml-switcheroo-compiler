"""Tests for test_types_utils."""

from __future__ import annotations

import ml_switcheroo_compiler.backends.eager as eager_pkg
import ml_switcheroo_compiler.backends.eager.types_utils as eager_types_utils
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import adapt_backend_module


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

    def array(
        self,
        data: OpValue,
        dtype: OpValue = None,
    ) -> OpValue:
        """Mock array.

        Args:
            data (OpValue): Input data.
            dtype (OpValue): Optional data type.

        Returns:
            OpValue: Mock array descriptor.
        """
        return ("array", data, dtype)

    def asarray(
        self,
        data: OpValue,
    ) -> OpValue:
        """Mock asarray.

        Args:
            data (OpValue): Input data.

        Returns:
            OpValue: Mock asarray descriptor.
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


class _MockNumpyObject:
    """Mock tensor-like object implementing numpy() without item()."""

    ndim: int = 0

    def numpy(self) -> _MockItemObject:
        """Return scalar wrapper exposing item().

        Returns:
            _MockItemObject: Mock scalar item holder.
        """
        return _MockItemObject()


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


def test_backends_eager_init_and_types_utils() -> None:
    """Verify eager package __init__ exports and all types_utils functions."""
    assert adapt_backend_module(eager_pkg).has_symbol("generic_zeros")
    assert adapt_backend_module(eager_pkg).has_symbol("generic_array")
    assert adapt_backend_module(eager_pkg).has_symbol("generic_asarray")
    assert adapt_backend_module(eager_pkg).has_symbol("generic_item")
    assert "elastic_transform_eager" in eager_pkg.__all__
    assert "_allclose" in eager_pkg.__all__

    mock_mod = _MockBackendModule()
    assert eager_types_utils.generic_zeros(mock_mod, (2, 3)) == ("zeros", (2, 3))
    assert eager_types_utils.generic_array(mock_mod, [1, 2]) == ("array", [1, 2], None)
    assert eager_types_utils.generic_array(mock_mod, [1, 2], dtype="float32") == ("array", [1, 2], "float32")
    assert eager_types_utils.generic_asarray(mock_mod, [3, 4]) == ("asarray", [3, 4])
    assert eager_types_utils.generic_item(mock_mod, _MockItemObject()) == 42.5
    assert eager_types_utils.generic_item(mock_mod, _MockNumpyObject()) == 42.5
