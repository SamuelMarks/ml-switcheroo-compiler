"""Tests for generalized_normal."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.generalized_normal import generalized_normal
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_generalized_normal() -> None:
    """Test generalized_normal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.generalized_normal"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = generalized_normal(1, 2, a=3)
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_dispatch.assert_called_once_with("generalized_normal", 1, 2, a=3)
