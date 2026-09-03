"""Tests for docs/conf.py."""

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs")))

import conf


@mock.patch("conf.setup_directive")
def test_setup(mock_setup_directive):
    """Test setup()."""
    app = mock.MagicMock()
    result = conf.setup(app)

    mock_setup_directive.assert_called_once_with(app)
    assert result is mock_setup_directive.return_value
