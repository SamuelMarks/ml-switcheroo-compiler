"""Tests for tensorflow types module."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.shape import _set_attr
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass

with patch.dict(sys.modules, {"tensorflow": MagicMock()}):
    from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros


def test_tf_types() -> None:
    """Test coverage for tensorflow types."""
    import ml_switcheroo_compiler.backends.tensorflow.types as mod

    _ = (zeros, array, asarray, item)
    _set_attr(mod, "generic_zeros", MagicMock(return_value="zeros"))
    _set_attr(mod, "generic_array", MagicMock(return_value="array"))
    _set_attr(mod, "generic_asarray", MagicMock(return_value="asarray"))
    _set_attr(mod, "generic_item", MagicMock(return_value="item"))

    assert as_callable(mod.zeros)(None, (2, 2)) == "zeros"
    assert as_callable(mod.array)(None, [1]) == "array"
    assert as_callable(mod.asarray)(None, [1]) == "asarray"
    assert as_callable(mod.item)(None, [1]) == "item"
