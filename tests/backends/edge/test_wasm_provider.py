"""Tests for test_wasm_provider."""

from __future__ import annotations

import typing
from pathlib import Path
from unittest import mock
from unittest.mock import mock_open, patch

import pytest
import yaml

import ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider as provider
from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers, get_js_orchestration_template, get_wasm_simd_op, get_wasm_simd_ops, get_wasm_template, load_yaml, load_yaml_dir
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


@pytest.fixture(autouse=True)
def reset_wasm_templates() -> Iterator[None]:
    """Test function reset_wasm_templates."""
    old = provider._WASM_TEMPLATES
    provider._WASM_TEMPLATES = {}
    yield
    provider._WASM_TEMPLATES = old


def test_wasm_provider_get_wasm_template_missing(mocker: EagerTensorProtocol) -> None:
    """Test wasm provider.

    Args:
        mocker: Test parameter `mocker`.
    """
    # To test not res (line 31)
    provider._WASM_TEMPLATES = {"templates": {}}
    assert get_wasm_template("not_a_template") == {}

    # To test templates not dict (line 35)
    provider._WASM_TEMPLATES = {"templates": []}
    assert get_wasm_template("something") == {}


def test_wasm_provider_get_js_orchestration_template(mocker: EagerTensorProtocol) -> None:
    """Test js orch.

    Args:
        mocker: Test parameter `mocker`.
    """
    provider._WASM_TEMPLATES = {}
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir", return_value={"js_orchestration": {"my_template": "hello"}})
    assert get_js_orchestration_template("my_template") == "hello"

    provider._WASM_TEMPLATES = {"js_orchestration": []}
    assert get_js_orchestration_template("my_template") == ""


def test_wasm_provider_get_cpp_helpers(mocker: EagerTensorProtocol) -> None:
    """Test cpp helpers.

    Args:
        mocker: Test parameter `mocker`.
    """
    provider._WASM_TEMPLATES = {}
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir", return_value={"cpp_helpers": ["help"]})
    assert get_cpp_helpers() == ["help"]

    provider._WASM_TEMPLATES = {"cpp_helpers": {}}
    assert get_cpp_helpers() == []


def test_load_yaml() -> None:
    """Test function test_load_yaml."""
    yaml_data = {"templates": {"test": {"body": "code"}}, "js_orchestration": {}, "cpp_helpers": []}
    with patch("builtins.open", mock_open(read_data=yaml.dump(yaml_data))):
        res = load_yaml("test.yaml")
        assert as_tensor(getattr(as_tensor(getattr(as_tensor(getattr(res, "__getitem__")("templates")), "__getitem__")("test")), "__getitem__")("body")) == "code"


def test_load_yaml_dir(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_load_yaml_dir.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_data1 = {"templates": {"t1": "string body", "t2": {"body": "dict body"}}}
    mock_data2 = {"templates": {"t3": ["list", "body"]}}
    mock_data3 = ["not", "a", "dict"]

    def mock_glob(path: EagerValue) -> EagerValue:
        """Test function mock_glob.

        Args:
            path: Test parameter `path`.
        """
        return ["file1.yaml", "file2.yaml", "file3.yaml"]

    def mock_open_impl(file: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_open_impl.

        Args:
            file: Test parameter `file`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        if file == "file1.yaml":
            return as_tensor(mock_open(read_data=yaml.dump(mock_data1))())
        elif file == "file2.yaml":
            return as_tensor(mock_open(read_data=yaml.dump(mock_data2))())
        else:
            return as_tensor(mock_open(read_data=yaml.dump(mock_data3))())

    with patch("glob.glob", side_effect=mock_glob):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("builtins.open", side_effect=mock_open_impl):
                res = load_yaml_dir("mock_dir")
                assert "t1" in res["templates"]
                assert as_tensor(getattr(as_tensor(getattr(res, "__getitem__")("templates")), "__getitem__")("t1")) == {"body": "string body"}
                assert "t2" in res["templates"]
                assert as_tensor(getattr(as_tensor(getattr(res, "__getitem__")("templates")), "__getitem__")("t2")) == {"body": "dict body"}
                assert "t3" in res["templates"]
                assert as_tensor(getattr(as_tensor(getattr(res, "__getitem__")("templates")), "__getitem__")("t3")) == ["list", "body"]


def test_load_yaml_dir_not_dir() -> None:
    """Test function test_load_yaml_dir_not_dir."""
    with patch("pathlib.Path.is_dir", return_value=False):
        res = load_yaml_dir("mock_dir")
        assert res == {"templates": {}}


def test_get_wasm_template_empty_global(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_get_wasm_template_empty_global.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    provider._WASM_TEMPLATES = {}
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir", return_value={"templates": {"my_tpl": {"body": "success"}}}):
        assert get_wasm_template("my_tpl") == {"body": "success"}
    provider._WASM_TEMPLATES = {"templates": {"my_tpl": {"body": "success"}}}
    assert get_wasm_template("my_tpl") == {"body": "success"}


def test_get_wasm_template_not_dict() -> None:
    """Test function test_get_wasm_template_not_dict."""
    setattr(provider, "_WASM_TEMPLATES", {"templates": {"my_tpl": ["not", "dict"]}})
    assert get_wasm_template("my_tpl") == {}


def test_wasm_simd_provider_branches(tmp_path: Path) -> None:
    """Test wasm_provider when simd_path does not exist or has non-dict data.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider as wp

    orig_simd_ops = dict(wp._WASM_SIMD_OPS)
    try:
        wp._WASM_SIMD_OPS.clear()
        tmp_path / "does_not_exist.yaml"
        with mock.patch.object(Path, "is_file", return_value=False):
            ops = get_wasm_simd_ops()
            assert isinstance(ops, dict)
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch("builtins.open", mock.mock_open(read_data="[1, 2, 3]")):
            wp._WASM_SIMD_OPS.clear()
            ops2 = get_wasm_simd_ops()
            assert isinstance(ops2, dict)
    finally:
        wp._WASM_SIMD_OPS = orig_simd_ops


def test_wasm_simd_ops_retrieval() -> None:
    """Test get_wasm_simd_ops and get_wasm_simd_op.

    Returns:
        None
    """
    ops = get_wasm_simd_ops()
    assert isinstance(ops, dict)
    add_op = get_wasm_simd_op("Add")
    assert isinstance(add_op, dict)
    missing_op = get_wasm_simd_op("NonExistentSimdOp")
    assert missing_op == {}
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_simd_ops", return_value={"operations": {"bad_op": "not_a_dict"}}):
        assert get_wasm_simd_op("bad_op") == {}
