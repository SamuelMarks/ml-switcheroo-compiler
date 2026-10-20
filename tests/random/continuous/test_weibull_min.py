"""Tests for weibull_min."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.weibull_min import weibull_min

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_weibull_min() -> None:
    """Test weibull_min function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.weibull_min"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = weibull_min(1, 2, a=3)
        assert typing.cast("EagerTensorProtocol", result) == typing.cast("EagerTensorProtocol", "mocked_result")
        mock_dispatch.assert_called_once_with("weibull_min", 1, 2, a=3)
