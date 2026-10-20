"""Tests for generalized_normal."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.generalized_normal import generalized_normal

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_generalized_normal() -> None:
    """Test generalized_normal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.generalized_normal"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = generalized_normal(1, 2, a=3)
        assert typing.cast("EagerTensorProtocol", result) == typing.cast("EagerTensorProtocol", "mocked_result")
        mock_dispatch.assert_called_once_with("generalized_normal", 1, 2, a=3)
