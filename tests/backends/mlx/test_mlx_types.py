"""Tests for test_mlx_types."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mlx_types() -> None:
    """Test function test_mlx_types."""
    import pytest

    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros

    res = cast("Callable[..., EagerTensorProtocol]", zeros)(None, (2, 2))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)

    res = cast("Callable[..., EagerTensorProtocol]", array)(None, [1.0, 2.0])
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)

    res = cast("Callable[..., EagerTensorProtocol]", asarray)(None, [1.0, 2.0])
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)

    res = cast("Callable[..., EagerTensorProtocol]", item)(None, cast("Callable[..., EagerTensorProtocol]", array)(None, 42.0))
    assert res == 42.0


def test_mlx_types_asarray_fallback(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_types_asarray_fallback."""
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMX:
        """Test class MockMX."""

        def array(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function array."""
            return "array_called"

    cast("Callable[..., EagerTensorProtocol]", monkeypatch.setattr)(types_mod, "mx", cast("Callable[..., EagerTensorProtocol]", MockMX)())
    res = cast("Callable[..., EagerTensorProtocol]", asarray)(None, [1.0])
    assert res == "array_called"


def test_mlx_types_asarray_success(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_types_asarray_success."""
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMXWithAsarray:
        """Test class MockMXWithAsarray."""

        def asarray(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function asarray."""
            return "asarray_called"

    cast("Callable[..., EagerTensorProtocol]", monkeypatch.setattr)(types_mod, "mx", cast("Callable[..., EagerTensorProtocol]", MockMXWithAsarray)())
    res = cast("Callable[..., EagerTensorProtocol]", asarray)(None, [1.0])
    assert res == "asarray_called"
