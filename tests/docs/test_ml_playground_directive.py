"""Tests for docs/ml_playground_directive.py."""

import os
import sys
from unittest import mock

from docutils import nodes  # type: ignore[import-untyped]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs")))

from ml_playground_directive import MLPlaygroundDirective, get_wheel_assets, setup


def test_ml_playground_directive_run():
    """Test MLPlaygroundDirective.run()."""
    directive = MLPlaygroundDirective(
        name="ml-playground",
        arguments=[],
        options={},
        content=[],
        lineno=1,
        content_offset=0,
        block_text=".. ml-playground::\n",
        state=mock.MagicMock(),
        state_machine=mock.MagicMock(),
    )
    result = directive.run()
    assert len(result) == 1
    assert isinstance(result[0], nodes.raw)
    raw_text = result[0].astext()
    assert 'id="ml-playground-container"' in raw_text
    assert 'class="pg-wheel-links"' in raw_text
    assert result[0]["format"] == "html"


def test_get_wheel_assets_missing_dir():
    """Test get_wheel_assets when _static directory does not exist."""
    with mock.patch("os.path.isdir", return_value=False):
        assert get_wheel_assets() == []


def test_get_wheel_assets_with_files(tmp_path):
    """Test get_wheel_assets with mock directory and files."""
    test_dir = tmp_path / "_static"
    test_dir.mkdir()
    (test_dir / "pkg_a-0.2.0-py3-none-any.whl").touch()
    (test_dir / "pkg_b-1.0.0-py3-none-any.whl").touch()
    (test_dir / "other.js").touch()

    with mock.patch("os.path.abspath", return_value=str(test_dir)), mock.patch("os.path.isdir", return_value=True):
        assets = get_wheel_assets()
        assert assets == ["pkg_a-0.2.0-py3-none-any.whl", "pkg_b-1.0.0-py3-none-any.whl"]
        assert (test_dir / "wheels.json").exists()


def test_get_wheel_assets_os_error(tmp_path):
    """Test get_wheel_assets handles OSError during manifest write."""
    test_dir = tmp_path / "_static"
    test_dir.mkdir()
    (test_dir / "pkg-0.1.0-py3-none-any.whl").touch()

    with mock.patch("os.path.abspath", return_value=str(test_dir)), mock.patch("os.path.isdir", return_value=True), mock.patch("builtins.open", side_effect=OSError("Permission denied")):
        assets = get_wheel_assets()
        assert assets == ["pkg-0.1.0-py3-none-any.whl"]


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
