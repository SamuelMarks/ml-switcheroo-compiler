"""Tests for wald."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.wald import wald
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_wald() -> None:
    """Test wald function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.wald"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = wald(1, 2, a=3)
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_dispatch.assert_called_once_with("wald", 1, 2, a=3)
