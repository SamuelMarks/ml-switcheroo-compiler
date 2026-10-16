"""Tests for rayleigh."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.rayleigh import rayleigh

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_rayleigh() -> None:
    """Test rayleigh function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.rayleigh"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = rayleigh(1, 2, a=3)
        assert cast("EagerTensorProtocol", result) == cast("EagerTensorProtocol", "mocked_result")
        mock_dispatch.assert_called_once_with("rayleigh", 1, 2, a=3)
