"""Tests for logistic."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.logistic import logistic


def test_logistic() -> None:
    """Test logistic function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.logistic"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = logistic(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("logistic", 1, 2, a=3)
