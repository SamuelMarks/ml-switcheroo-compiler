"""Tests for docs/ml_playground_directive.py."""

import os
import sys
from unittest import mock

from docutils import nodes  # type: ignore[import-untyped]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs")))

from ml_playground_directive import MLPlaygroundDirective, setup


def test_ml_playground_directive_run():
    """Test MLPlaygroundDirective.run()."""
    directive = MLPlaygroundDirective(name="ml-playground", arguments=[], options={}, content=[], lineno=1, content_offset=0, block_text=".. ml-playground::\n", state=mock.MagicMock(), state_machine=mock.MagicMock())
    result = directive.run()
    assert len(result) == 1
    assert isinstance(result[0], nodes.raw)
    assert 'id="ml-playground-container"' in result[0].astext()
    assert result[0]["format"] == "html"


def test_setup():
    """Test setup()."""
    app = mock.MagicMock()
    result = setup(app)

    app.add_directive.assert_called_once_with("ml-playground", MLPlaygroundDirective)
    app.add_css_file.assert_any_call("playground.css")
    app.add_js_file.assert_any_call("playground.js")
    app.add_js_file.assert_any_call("webgpu_runner.js")
    app.add_js_file.assert_any_call("wasm_runner.js")

    assert result == {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
