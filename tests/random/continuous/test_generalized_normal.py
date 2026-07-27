"""Tests for generalized_normal."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.generalized_normal import generalized_normal


def test_generalized_normal() -> None:
    """Test generalized_normal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.generalized_normal"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = generalized_normal(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("generalized_normal", 1, 2, a=3)
