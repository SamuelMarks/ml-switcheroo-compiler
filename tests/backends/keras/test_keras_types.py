"""Test module."""

from __future__ import annotations

import typing
from unittest.mock import patch

from ml_switcheroo_compiler.backends.keras.types import array, asarray, item, zeros

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_keras_types() -> None:
    """Test function test_keras_types."""
    with patch("ml_switcheroo_compiler.backends.keras.types.kops"):
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_zeros", return_value="zeros"):
            assert typing.cast("EagerCallableProtocol", zeros)(None, (2,)) == "zeros"
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_array", return_value="array"):
            assert typing.cast("EagerCallableProtocol", array)(None, [1]) == "array"
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_asarray", return_value="asarray"):
            assert typing.cast("EagerCallableProtocol", asarray)(None, [1]) == "asarray"
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_item", return_value=42.0):
            assert typing.cast("EagerCallableProtocol", item)(None, [1]) == 42.0
