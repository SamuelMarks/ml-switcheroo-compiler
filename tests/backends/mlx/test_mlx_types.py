"""Tests for test_mlx_types."""

from __future__ import annotations

import typing

from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mlx_types() -> None:
    """Test function test_mlx_types."""
    import pytest

    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")

    from ml_switcheroo_compiler.backends.mlx.types import array, asarray, item, zeros

    res = as_callable(zeros)(None, (2, 2))
    assert as_tensor(getattr(res, "shape")) == (2, 2)

    res = as_callable(array)(None, [1.0, 2.0])
    assert as_tensor(getattr(res, "shape")) == (2,)

    res = as_callable(asarray)(None, [1.0, 2.0])
    assert as_tensor(getattr(res, "shape")) == (2,)

    res = as_callable(item)(None, as_callable(array)(None, 42.0))
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

    as_callable(monkeypatch.setattr)(types_mod, "mx", as_callable(MockMX)())
    res = as_callable(asarray)(None, [1.0])
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

    as_callable(monkeypatch.setattr)(types_mod, "mx", as_callable(MockMXWithAsarray)())
    res = as_callable(asarray)(None, [1.0])
    assert res == "asarray_called"
