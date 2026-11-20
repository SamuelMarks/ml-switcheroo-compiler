"""Tests for test_wasm_provider."""

from __future__ import annotations

import io
import typing
from pathlib import Path
from unittest import mock
from unittest.mock import mock_open, patch

import pytest
import yaml
from pytest_mock import MockerFixture

import ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider as provider
from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import (
    get_cpp_helpers,
    get_js_orchestration_template,
    get_wasm_simd_op,
    get_wasm_simd_ops,
    get_wasm_template,
    load_yaml,
    load_yaml_dir,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


@pytest.fixture(autouse=True)
def reset_wasm_templates() -> Iterator[None]:
    """Test function reset_wasm_templates.

    Yields:
        None: Yielded test context or value from `reset_wasm_templates`.
    """
    old = provider._WASM_TEMPLATES
    provider._WASM_TEMPLATES = {}
    yield
    provider._WASM_TEMPLATES = old


def test_wasm_provider_get_wasm_template_missing(mocker: MockerFixture) -> None:
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


def test_wasm_provider_get_js_orchestration_template(mocker: MockerFixture) -> None:
    """Test js orch.

    Args:
        mocker: Test parameter `mocker`.
    """
    provider._WASM_TEMPLATES = {}
    mocker.patch(
        "ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir",
        return_value={"js_orchestration": {"my_template": "hello"}},
    )
    assert get_js_orchestration_template("my_template") == "hello"

    provider._WASM_TEMPLATES = {"js_orchestration": []}
    assert get_js_orchestration_template("my_template") == ""


def test_wasm_provider_get_cpp_helpers(mocker: MockerFixture) -> None:
    """Test cpp helpers.

    Args:
        mocker: Test parameter `mocker`.
    """
    provider._WASM_TEMPLATES = {}
    mocker.patch(
        "ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir",
        return_value={"cpp_helpers": ["help"]},
    )
    assert get_cpp_helpers() == ["help"]

    provider._WASM_TEMPLATES = {"cpp_helpers": {}}
    assert get_cpp_helpers() == []


def test_load_yaml() -> None:
    """Test function test_load_yaml."""
    yaml_data = {"templates": {"test": {"body": "code"}}, "js_orchestration": {}, "cpp_helpers": []}
    with patch("builtins.open", mock_open(read_data=yaml.dump(yaml_data))):
        res = load_yaml("test.yaml")
        templates = res["templates"]
        assert isinstance(templates, typing.Mapping)
        test_tpl = templates["test"]
        assert isinstance(test_tpl, typing.Mapping)
        assert test_tpl["body"] == "code"


def test_load_yaml_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_load_yaml_dir.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    mock_data1 = {"templates": {"t1": "string body", "t2": {"body": "dict body"}}}
    mock_data2 = {"templates": {"t3": ["list", "body"]}}
    mock_data3 = ["not", "a", "dict"]

    def mock_glob(path: BackendArg) -> BackendArg:
        """Test function mock_glob.

        Args:
            path: Test parameter `path`.

        Returns:
            BackendArg: Computed output value from `mock_glob`.
        """
        return ["file1.yaml", "file2.yaml", "file3.yaml"]

    def mock_open_impl(file: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> io.StringIO:
        """Test function mock_open_impl.

        Args:
            file: Test parameter `file`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            io.StringIO: Computed output value from `mock_open_impl`.
        """
        if file == "file1.yaml":
            return io.StringIO(yaml.dump(mock_data1))
        elif file == "file2.yaml":
            return io.StringIO(yaml.dump(mock_data2))
        else:
            return io.StringIO(yaml.dump(mock_data3))

    with patch("glob.glob", side_effect=mock_glob):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("builtins.open", side_effect=mock_open_impl):
                res = load_yaml_dir("mock_dir")
                templates = res["templates"]
                assert isinstance(templates, typing.Mapping)
                assert "t1" in templates
                assert templates["t1"] == {"body": "string body"}
                assert "t2" in templates
                assert templates["t2"] == {"body": "dict body"}
                assert "t3" in templates
                t3_val: BackendArg = templates["t3"]
                assert t3_val == ["list", "body"]


def test_load_yaml_dir_not_dir() -> None:
    """Test function test_load_yaml_dir_not_dir."""
    with patch("pathlib.Path.is_dir", return_value=False):
        res = load_yaml_dir("mock_dir")
        assert res == {"templates": {}}


def test_get_wasm_template_empty_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_get_wasm_template_empty_global.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    provider._WASM_TEMPLATES = {}
    with patch(
        "ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml_dir",
        return_value={"templates": {"my_tpl": {"body": "success"}}},
    ):
        assert get_wasm_template("my_tpl") == {"body": "success"}
    provider._WASM_TEMPLATES = {"templates": {"my_tpl": {"body": "success"}}}
    assert get_wasm_template("my_tpl") == {"body": "success"}


def test_get_wasm_template_not_dict() -> None:
    """Test function test_get_wasm_template_not_dict."""
    pytest.MonkeyPatch().setattr(provider, "_WASM_TEMPLATES", {"templates": {"my_tpl": ["not", "dict"]}}, raising=False)
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
        with (
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch("builtins.open", mock.mock_open(read_data="[1, 2, 3]")),
        ):
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
    with patch(
        "ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_simd_ops",
        return_value={"operations": {"bad_op": "not_a_dict"}},
    ):
        assert get_wasm_simd_op("bad_op") == {}
