"""Tests for orthogonal."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.orthogonal import orthogonal


def test_orthogonal() -> None:
    """Test orthogonal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.orthogonal"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = orthogonal(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("orthogonal", 1, 2, a=3)
