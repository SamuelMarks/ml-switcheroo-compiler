"""Tests for loggamma."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.loggamma import loggamma


def test_loggamma() -> None:
    """Test loggamma function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.loggamma"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = loggamma(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("loggamma", 1, 2, a=3)
