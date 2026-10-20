"""Tests for random_gamma_p."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.random_gamma_p import random_gamma_p

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_random_gamma_p() -> None:
    """Test random_gamma_p function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.random_gamma_p"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = random_gamma_p(1, 2, a=3)
        assert typing.cast("EagerTensorProtocol", result) == typing.cast("EagerTensorProtocol", "mocked_result")
        mock_dispatch.assert_called_once_with("random_gamma_p", 1, 2, a=3)
