"""Tests for t."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.t import t


def test_t() -> None:
    """Test t function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.t"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = t(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("t", 1, 2, a=3)
