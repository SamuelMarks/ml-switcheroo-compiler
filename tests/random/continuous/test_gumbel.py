"""Tests for gumbel."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.gumbel import gumbel


def test_gumbel() -> None:
    """Test gumbel function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.gumbel"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = gumbel(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("gumbel", 1, 2, a=3)
