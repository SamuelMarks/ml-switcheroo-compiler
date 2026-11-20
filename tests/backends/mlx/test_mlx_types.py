"""Tests for test_mlx_types."""

from __future__ import annotations

import typing

import pytest

from tests.testing_utils import extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_mlx_types() -> None:
    """Test function test_mlx_types."""
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros

    res = zeros(None, (2, 2))
    assert extract_test_shape(res) == (2, 2)

    res = array(None, [1.0, 2.0])
    assert extract_test_shape(res) == (2,)

    res = asarray(None, [1.0, 2.0])
    assert extract_test_shape(res) == (2,)

    res = item(None, array(None, 42.0))
    assert res == 42.0


def test_mlx_types_asarray_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_mlx_types_asarray_fallback.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMX:
        """Test class MockMX."""

        def array(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return "array_called"

    monkeypatch.setattr(types_mod, "mx", MockMX())
    res = asarray(None, [1.0])
    assert res == "array_called"


def test_mlx_types_asarray_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_mlx_types_asarray_success.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMXWithAsarray:
        """Test class MockMXWithAsarray."""

        def asarray(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return "asarray_called"

    monkeypatch.setattr(types_mod, "mx", MockMXWithAsarray())
    res = asarray(None, [1.0])
    assert res == "asarray_called"
