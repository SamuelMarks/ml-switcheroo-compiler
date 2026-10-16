"""Tests for docs/conf.py."""

import importlib
import os
import sys
from unittest import mock

import pytest

pytest.importorskip("docutils")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs")))
conf = importlib.import_module("conf")


@mock.patch("conf.setup_directive")
def test_setup(mock_setup_directive: mock.MagicMock) -> None:
    """Test setup().

    Args:
        mock_setup_directive (mock.MagicMock): Mocked setup_directive callable.
    """
    app = mock.MagicMock()
    result = conf.setup(app)

    mock_setup_directive.assert_called_once_with(app)
    assert result is mock_setup_directive.return_value
