"""Tests for random_gamma_p."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.random_gamma_p import random_gamma_p


def test_random_gamma_p() -> None:
    """Test random_gamma_p function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.random_gamma_p"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = random_gamma_p(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("random_gamma_p", 1, 2, a=3)
