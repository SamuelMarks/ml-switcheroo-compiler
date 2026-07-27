"""Tests for chisquare."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.chisquare import chisquare


def test_chisquare() -> None:
    """Test chisquare function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.chisquare"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = chisquare(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("chisquare", 1, 2, a=3)
