"""Tests for double_sided_maxwell."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.double_sided_maxwell import double_sided_maxwell


def test_double_sided_maxwell() -> None:
    """Test double_sided_maxwell function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.double_sided_maxwell"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = double_sided_maxwell(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("double_sided_maxwell", 1, 2, a=3)
