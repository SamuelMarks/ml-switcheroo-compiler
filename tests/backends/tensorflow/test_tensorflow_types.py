"""Tests for tensorflow types module."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo_compiler.ops as _ops_pkg

if typing.TYPE_CHECKING:
    pass


def test_tf_types() -> None:
    """Test coverage for tensorflow types."""
    assert _ops_pkg is not None
    with patch.dict(sys.modules, {"tensorflow": MagicMock()}):
        import ml_switcheroo_compiler.backends.tensorflow.types as mod
        from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros

        _ = (zeros, array, asarray, item)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod, "generic_zeros", MagicMock(return_value="zeros"), raising=False)
            mp.setattr(mod, "generic_array", MagicMock(return_value="array"), raising=False)
            mp.setattr(mod, "generic_asarray", MagicMock(return_value="asarray"), raising=False)
            mp.setattr(mod, "generic_item", MagicMock(return_value=1.0), raising=False)

            assert mod.zeros(None, (2, 2)) == "zeros"
            assert mod.array(None, [1]) == "array"
            assert mod.asarray(None, [1]) == "asarray"
            assert mod.item(None, [1]) == 1.0
