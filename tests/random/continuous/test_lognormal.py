"""Tests for lognormal."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.lognormal import lognormal

if typing.TYPE_CHECKING:
    pass


def test_lognormal() -> None:
    """Test lognormal function."""
    with patch.object(
        sys.modules["ml_switcheroo_compiler.random.continuous.lognormal"], "_dispatch_random"
    ) as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = lognormal(1, 2, a=3)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("lognormal", 1, 2, a=3)
