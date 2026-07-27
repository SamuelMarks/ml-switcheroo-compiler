"""Tests for tensorflow types module."""

import sys
from unittest.mock import MagicMock

mock_tf = MagicMock()
sys.modules["tensorflow"] = mock_tf

from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros


def test_tf_types() -> None:
    """Test coverage for tensorflow types."""
    import ml_switcheroo_compiler.backends.tensorflow.types as mod

    mod.generic_zeros = MagicMock(return_value="zeros")
    mod.generic_array = MagicMock(return_value="array")
    mod.generic_asarray = MagicMock(return_value="asarray")
    mod.generic_item = MagicMock(return_value="item")

    assert zeros(None, (2, 2)) == "zeros"
    assert array(None, [1]) == "array"
    assert asarray(None, [1]) == "asarray"
    assert item(None, [1]) == "item"
