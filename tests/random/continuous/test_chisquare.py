"""Tests for chisquare."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.chisquare import chisquare

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_chisquare() -> None:
    """Test chisquare function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.chisquare"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = chisquare(1, 2, a=3)
        assert typing.cast("EagerTensorProtocol", result) == typing.cast("EagerTensorProtocol", "mocked_result")
        mock_dispatch.assert_called_once_with("chisquare", 1, 2, a=3)
