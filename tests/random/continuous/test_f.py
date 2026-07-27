"""Tests for f."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.f import f


def test_f() -> None:
    """Test f function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.f"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = f(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("f", 1, 2, a=3)
