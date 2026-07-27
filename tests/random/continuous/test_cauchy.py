"""Tests for cauchy."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.cauchy import cauchy


def test_cauchy() -> None:
    """Test cauchy function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.cauchy"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = cauchy(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("cauchy", 1, 2, a=3)
