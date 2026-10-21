"""Tests for cauchy."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

from ml_switcheroo_compiler.random.continuous.cauchy import cauchy
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_cauchy() -> None:
    """Test cauchy function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.continuous.cauchy"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = cauchy(1, 2, a=3)
        assert as_tensor(result) == as_tensor("mocked_result")
        mock_dispatch.assert_called_once_with("cauchy", 1, 2, a=3)
