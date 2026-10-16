"""Tests for test_mlx_types."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mlx_asarray_fallback(mocker: EagerTensorProtocol) -> None:
    """Test mlx asarray fallback."""
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class DummyMX:
        """Test class DummyMX."""

        def array(self, data: EagerValue) -> EagerValue:
            """Test function array."""
            return "array_" + str(data)

    cast("Callable[..., EagerTensorProtocol]", mocker.patch.object)(types_mod, "mx", cast("Callable[..., EagerTensorProtocol]", DummyMX)())

    assert cast("Callable[..., EagerTensorProtocol]", asarray)(None, "foo") == "array_foo"
