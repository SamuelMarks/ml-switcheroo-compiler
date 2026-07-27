"""Tests for maxwell."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.maxwell import maxwell


def test_maxwell() -> None:
    """Test maxwell function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.maxwell"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = maxwell(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("maxwell", 1, 2, a=3)
