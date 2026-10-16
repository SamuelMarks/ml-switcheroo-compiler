"""Tests for laplace."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.laplace import laplace

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_laplace() -> None:
    """Test laplace function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.laplace"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = laplace(1, 2, a=3)
        assert cast("EagerTensorProtocol", result) == cast("EagerTensorProtocol", "mocked_result")
        mock_dispatch.assert_called_once_with("laplace", 1, 2, a=3)
