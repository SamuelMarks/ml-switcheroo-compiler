"""Tests for wald."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.wald import wald


def test_wald() -> None:
    """Test wald function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.wald"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = wald(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("wald", 1, 2, a=3)
