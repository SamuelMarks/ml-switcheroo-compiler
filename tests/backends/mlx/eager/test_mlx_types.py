"""Tests for test_mlx_types."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_mlx_asarray_fallback(mocker: EagerTensorProtocol) -> None:
    """Test mlx asarray fallback.

    Args:
        mocker: Test parameter `mocker`.
    """
    import ml_switcheroo_compiler.backends.mlx.types as types_mod
    from ml_switcheroo_compiler.backends.mlx.types import asarray

    class DummyMX:
        """Test class DummyMX."""

        def array(self, data: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                data: Test parameter `data`.
            """
            return "array_" + str(data)

    typing.cast("EagerCallableProtocol", mocker.patch.object)(types_mod, "mx", typing.cast("EagerCallableProtocol", DummyMX)())

    assert typing.cast("EagerCallableProtocol", asarray)(None, "foo") == "array_foo"
