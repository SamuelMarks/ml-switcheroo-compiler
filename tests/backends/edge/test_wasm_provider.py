from unittest.mock import mock_open, patch

import pytest
import yaml

import ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider as provider
from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers, get_js_orchestration_template, get_wasm_template, load_yaml, load_yaml_dir


@pytest.fixture(autouse=True)
def reset_wasm_templates():
    old = provider._WASM_TEMPLATES
    provider._WASM_TEMPLATES = {}
    yield
    provider._WASM_TEMPLATES = old


def test_wasm_provider_get_wasm_template_missing(mocker):
    """Test wasm provider."""
    # To test not res (line 31)
    provider._WASM_TEMPLATES = {"templates": {}}
    assert get_wasm_template("not_a_template") == {}

    # To test templates not dict (line 35)
    provider._WASM_TEMPLATES = {"templates": []}
    assert get_wasm_template("something") == {}


def test_wasm_provider_get_js_orchestration_template(mocker):
    """Test js orch."""
    provider._WASM_TEMPLATES = {}
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir", return_value={"js_orchestration": {"my_template": "hello"}})
    assert get_js_orchestration_template("my_template") == "hello"

    provider._WASM_TEMPLATES = {"js_orchestration": []}
    assert get_js_orchestration_template("my_template") == ""


def test_wasm_provider_get_cpp_helpers(mocker):
    """Test cpp helpers."""
    provider._WASM_TEMPLATES = {}
    mocker.patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir", return_value={"cpp_helpers": ["help"]})
    assert get_cpp_helpers() == ["help"]

    provider._WASM_TEMPLATES = {"cpp_helpers": {}}
    assert get_cpp_helpers() == []


def test_load_yaml():
    yaml_data = {"templates": {"test": {"body": "code"}}, "js_orchestration": {}, "cpp_helpers": []}
    with patch("builtins.open", mock_open(read_data=yaml.dump(yaml_data))):
        res = load_yaml("test.yaml")
        assert res["templates"]["test"]["body"] == "code"


def test_load_yaml_dir(monkeypatch):
    mock_data1 = {"templates": {"t1": "string body", "t2": {"body": "dict body"}}}
    mock_data2 = {"templates": {"t3": ["list", "body"]}}

    def mock_glob(path):
        return ["file1.yaml", "file2.yaml"]

    def mock_open_impl(file, *args, **kwargs):
        if file == "file1.yaml":
            return mock_open(read_data=yaml.dump(mock_data1))()
        else:
            return mock_open(read_data=yaml.dump(mock_data2))()

    with patch("glob.glob", side_effect=mock_glob):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("builtins.open", side_effect=mock_open_impl):
                res = load_yaml_dir("mock_dir")
                assert "t1" in res["templates"]
                assert res["templates"]["t1"] == {"body": "string body"}
                assert "t2" in res["templates"]
                assert res["templates"]["t2"] == {"body": "dict body"}
                assert "t3" in res["templates"]
                assert res["templates"]["t3"] == ["list", "body"]


def test_load_yaml_dir_not_dir():
    with patch("pathlib.Path.is_dir", return_value=False):
        res = load_yaml_dir("mock_dir")
        assert res == {"templates": {}}


def test_get_wasm_template_empty_global(monkeypatch):
    provider._WASM_TEMPLATES = {}
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir", return_value={"templates": {"my_tpl": {"body": "success"}}}):
        assert get_wasm_template("my_tpl") == {"body": "success"}
    provider._WASM_TEMPLATES = {"templates": {"my_tpl": {"body": "success"}}}
    assert get_wasm_template("my_tpl") == {"body": "success"}


def test_get_wasm_template_not_dict():
    provider._WASM_TEMPLATES = {"templates": {"my_tpl": ["not", "dict"]}}
    assert get_wasm_template("my_tpl") == {}
