"""Tests for triangular."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.triangular import triangular


def test_triangular() -> None:
    """Test triangular function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.triangular"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = triangular(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("triangular", 1, 2, a=3)
