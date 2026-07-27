"""Tests for pareto."""

import sys
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.pareto import pareto


def test_pareto() -> None:
    """Test pareto function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.pareto"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = pareto(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("pareto", 1, 2, a=3)
