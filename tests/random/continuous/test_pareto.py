"""Tests for pareto."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.pareto import pareto
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_pareto() -> None:
    """Test pareto function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.pareto"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = pareto(1, 2, a=3)
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_dispatch.assert_called_once_with("pareto", 1, 2, a=3)
