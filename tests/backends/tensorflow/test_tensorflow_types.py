"""Tests for tensorflow types module."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol

with patch.dict(sys.modules, {"tensorflow": MagicMock()}):
    from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros


def test_tf_types() -> None:
    """Test coverage for tensorflow types."""
    import ml_switcheroo_compiler.backends.tensorflow.types as mod

    setattr(mod, "generic_zeros", MagicMock(return_value="zeros"))
    setattr(mod, "generic_array", MagicMock(return_value="array"))
    setattr(mod, "generic_asarray", MagicMock(return_value="asarray"))
    setattr(mod, "generic_item", MagicMock(return_value="item"))

    assert typing.cast("EagerCallableProtocol", zeros)(None, (2, 2)) == "zeros"
    assert typing.cast("EagerCallableProtocol", array)(None, [1]) == "array"
    assert typing.cast("EagerCallableProtocol", asarray)(None, [1]) == "asarray"
    assert typing.cast("EagerCallableProtocol", item)(None, [1]) == "item"
