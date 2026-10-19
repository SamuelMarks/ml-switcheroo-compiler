"""Tests for test_mlx_types."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_mlx_types() -> None:
    """Test function test_mlx_types."""
    import pytest

    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros

    res = cast("EagerCallableProtocol", zeros)(None, (2, 2))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)

    res = cast("EagerCallableProtocol", array)(None, [1.0, 2.0])
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)

    res = cast("EagerCallableProtocol", asarray)(None, [1.0, 2.0])
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)

    res = cast("EagerCallableProtocol", item)(None, cast("EagerCallableProtocol", array)(None, 42.0))
    assert res == 42.0


def test_mlx_types_asarray_fallback(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_types_asarray_fallback.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMX:
        """Test class MockMX."""

        def array(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "array_called"

    cast("EagerCallableProtocol", monkeypatch.setattr)(types_mod, "mx", cast("EagerCallableProtocol", MockMX)())
    res = cast("EagerCallableProtocol", asarray)(None, [1.0])
    assert res == "array_called"


def test_mlx_types_asarray_success(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_mlx_types_asarray_success.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class MockMXWithAsarray:
        """Test class MockMXWithAsarray."""

        def asarray(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function asarray.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "asarray_called"

    cast("EagerCallableProtocol", monkeypatch.setattr)(types_mod, "mx", cast("EagerCallableProtocol", MockMXWithAsarray)())
    res = cast("EagerCallableProtocol", asarray)(None, [1.0])
    assert res == "asarray_called"
