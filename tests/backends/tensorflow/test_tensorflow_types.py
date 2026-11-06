"""Tests for tensorflow types module."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    pass

with patch.dict(sys.modules, {"tensorflow": MagicMock()}):
    from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros


def test_tf_types() -> None:
    """Test coverage for tensorflow types."""
    import ml_switcheroo_compiler.backends.tensorflow.types as mod

    _ = (zeros, array, asarray, item)
    ensure_mutable(mod).generic_zeros = ensure_eager_tensor(MagicMock(return_value="zeros"))
    ensure_mutable(mod).generic_array = ensure_eager_tensor(MagicMock(return_value="array"))
    ensure_mutable(mod).generic_asarray = ensure_eager_tensor(MagicMock(return_value="asarray"))
    ensure_mutable(mod).generic_item = ensure_eager_tensor(MagicMock(return_value="item"))

    assert ensure_callable(mod.zeros)(None, (2, 2)) == "zeros"
    assert ensure_callable(mod.array)(None, [1]) == "array"
    assert ensure_callable(mod.asarray)(None, [1]) == "asarray"
    assert ensure_callable(mod.item)(None, [1]) == "item"
