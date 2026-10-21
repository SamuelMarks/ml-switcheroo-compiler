"""Tests for t."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.t import t
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_t() -> None:
    """Test t function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.t"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = t(1, 2, a=3)
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_dispatch.assert_called_once_with("t", 1, 2, a=3)
