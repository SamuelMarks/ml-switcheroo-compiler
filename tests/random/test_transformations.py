"""Tests for transformations."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.random.transformations import shuffle
from tests.testing_utils import as_callable, as_tensor, as_tensor_obj

if typing.TYPE_CHECKING:
    pass


def test_shuffle() -> None:
    """Test shuffle function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.transformations"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        x_mock = MagicMock()
        x_mock.shape = (5,)
        x_mock.dtype = dtypes.DType.Float32
        result = shuffle("key", x_mock, axis=1)
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_emit.assert_called_once_with("RandomShuffle", ["key", x_mock], (5,), dtypes.DType.Float32, {"axis": 1})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.transformations"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = as_tensor_obj(as_callable(shuffle)("key", "x"))
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_emit.assert_called_once_with("RandomShuffle", ["key", "x"], (), None, {"axis": 0})
