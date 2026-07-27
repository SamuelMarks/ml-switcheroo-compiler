"""Tests for rayleigh."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.rayleigh import rayleigh


def test_rayleigh() -> None:
    """Test rayleigh function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.rayleigh"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = rayleigh(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("rayleigh", 1, 2, a=3)
