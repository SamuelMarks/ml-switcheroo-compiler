"""Tests for orthogonal."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.orthogonal import orthogonal
from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_orthogonal() -> None:
    """Test orthogonal function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.orthogonal"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = orthogonal(1, 2, a=3)
        assert ensure_eager_tensor(result) == ensure_eager_tensor("mocked_result")
        mock_dispatch.assert_called_once_with("orthogonal", 1, 2, a=3)
