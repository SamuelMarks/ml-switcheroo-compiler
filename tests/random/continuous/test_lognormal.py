"""Tests for lognormal."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.lognormal import lognormal


def test_lognormal() -> None:
    """Test lognormal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.lognormal"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = lognormal(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("lognormal", 1, 2, a=3)
