"""Tests for exponential."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.exponential import exponential


def test_exponential() -> None:
    """Test exponential function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.exponential"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = exponential(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("exponential", 1, 2, a=3)
