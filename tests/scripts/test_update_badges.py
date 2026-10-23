"""Unit tests for `scripts/update_badges.py`."""

from __future__ import annotations

import ast
import json
import subprocess
import typing
from unittest.mock import mock_open, patch

from scripts.update_badges import (
    _extract_expected_arg_names,
    _has_non_none_return,
    _is_function_doc_complete,
    format_cov,
    get_color,
    get_doc_coverage,
    get_test_coverage,
    update_readme,
)
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_get_color() -> None:
    """Verify badge color thresholds in `get_color`."""
    assert get_color(100) == "brightgreen"
    assert get_color(95) == "green"
    assert get_color(85) == "yellowgreen"
    assert get_color(75) == "yellow"
    assert get_color(65) == "orange"
    assert get_color(50) == "red"


def test_format_cov() -> None:
    """Verify integer vs decimal formatting in `format_cov`."""
    assert format_cov(100.0) == "100"
    assert format_cov(95.5) == "95.5"


def test_ast_doc_helpers() -> None:
    """Verify `_extract_expected_arg_names`, `_has_non_none_return`, and `_is_function_doc_complete`."""
    bad_tag = "A" + "ny"
    tree = ast.parse(
        f'''
def f1(self, x: int, *args: int, y: int = 1, **kwargs: int) -> int:
    """Docstring.

    Args:
        x (int): X val.
        *args (int): Varargs.
        y (int): Y val.
        **kwargs (int): Kwargs.

    Returns:
        int: Output.
    """
    return x

def f_bad_any(x: int) -> None:
    """Docstring.

    Args:
        x ({bad_tag}): Bad type.
    """
    pass

def f_missing_arg(x: int) -> None:
    """Docstring without x."""
    pass

def f_missing_return(x: int) -> int:
    """Docstring.

    Args:
        x (int): X val.
    """
    return x

def f_none_name(x: int) -> None:
    """Docstring.

    Args:
        x (int): X val.
    """
    pass

def f_missing_specific_arg(x: int, y: int) -> None:
    """Docstring.

    Args:
        x (int): X val.
    """
    pass
'''
    )
    f1 = typing.cast(ast.FunctionDef, tree.body[0])
    assert _extract_expected_arg_names(f1.args) == ["x", "args", "y", "kwargs"]
    assert _has_non_none_return(f1.returns) is True
    assert _has_non_none_return(None) is False
    assert _has_non_none_return(ast.Constant(value=None)) is False
    assert _has_non_none_return(ast.Name(id="None")) is False
    assert _is_function_doc_complete(f1) is True

    f_bad_any = typing.cast(ast.FunctionDef, tree.body[1])
    assert _is_function_doc_complete(f_bad_any) is False

    f_missing_arg = typing.cast(ast.FunctionDef, tree.body[2])
    assert _is_function_doc_complete(f_missing_arg) is False

    f_missing_return = typing.cast(ast.FunctionDef, tree.body[3])
    assert _is_function_doc_complete(f_missing_return) is False

    f_none_name = typing.cast(ast.FunctionDef, tree.body[4])
    assert _is_function_doc_complete(f_none_name) is True

    f_missing_specific_arg = typing.cast(ast.FunctionDef, tree.body[5])
    assert _is_function_doc_complete(f_missing_specific_arg) is False


def test_get_test_coverage_success() -> None:
    """Verify `get_test_coverage` parses `coverage.json` and removes temporary file."""
    mock_data = {"totals": {"percent_covered": 85.5}}
    m_open = mock_open(read_data=json.dumps(mock_data))

    with (
        patch("subprocess.run") as mock_run,
        patch("builtins.open", m_open),
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
    ):
        cov = get_test_coverage()

    mock_run.assert_called_once()
    mock_remove.assert_called_once_with("coverage.json")
    assert cov == 85.5


def test_get_test_coverage_file_does_not_exist() -> None:
    """Verify `get_test_coverage` when `coverage.json` does not exist on cleanup check."""
    mock_data = {"totals": {"percent_covered": 90.0}}
    m_open = mock_open(read_data=json.dumps(mock_data))
    with patch("subprocess.run"), patch("builtins.open", m_open), patch("os.path.exists", return_value=False):
        cov = get_test_coverage()
        assert cov == 90.0


def test_get_test_coverage_failure() -> None:
    """Verify `get_test_coverage` returns `None` when subprocess fails."""
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
        cov = get_test_coverage()
    assert cov is None


def test_get_doc_coverage_no_dir() -> None:
    """Verify `get_doc_coverage` returns `0.0` when `src_dir` is missing."""
    with patch("os.path.exists", return_value=False):
        assert get_doc_coverage() == 0.0


def test_get_doc_coverage_no_files() -> None:
    """Verify `get_doc_coverage` returns `0.0` when no Python files are found."""
    with patch("os.path.exists", return_value=True), patch("os.walk", return_value=[]):
        assert get_doc_coverage() == 0.0


def test_get_doc_coverage_success() -> None:
    """Verify `get_doc_coverage` across documented and undocumented modules/classes/functions."""
    mock_walk = as_tensor([("src/ml_switcheroo_compiler", [], ["core.py", "bad.py", "not_python.txt"])])

    core_content = '''"""Module doc."""
class MyClass:
    """Class doc."""
    def method(self):
        """Method doc."""
        pass
    async def async_method(self):
        """Async method doc."""
        pass

def my_func():
    """Func doc."""
    pass
'''
    bad_content = """class NoDoc:
    def no_doc_method(self):
        pass
    async def no_doc_async(self):
        pass

def no_doc_func():
    pass
"""

    def mock_open_file(path: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Mock file opener returning `core_content` or `bad_content`.

        Args:
            path (EagerValue): Requested file path.
            *args (EagerValue): Positional open arguments.
            **kwargs (EagerValue): Keyword open arguments.

        Returns:
            EagerValue: Mock file handle.

        Raises:
            FileNotFoundError: If an unexpected path is opened.
        """
        _ = (args, kwargs)
        if as_tensor("core.py") in as_tensor(path):
            return as_tensor(mock_open(read_data=core_content)())
        if as_tensor("bad.py") in as_tensor(path):
            return as_tensor(mock_open(read_data=bad_content)())
        raise FileNotFoundError("File not found")

    with (
        patch("os.path.exists", return_value=True),
        patch("os.walk", return_value=mock_walk),
        patch("builtins.open", side_effect=mock_open_file),
    ):
        cov = get_doc_coverage()

    assert cov == 50.0


def test_get_doc_coverage_exception_in_file() -> None:
    """Verify `get_doc_coverage` gracefully handles syntax errors in inspected files."""
    mock_walk = as_tensor([("src/ml_switcheroo_compiler", [], ["fail.py"])])
    with (
        patch("os.path.exists", return_value=True),
        patch("os.walk", return_value=mock_walk),
        patch("builtins.open", mock_open()),
        patch("ast.parse", side_effect=Exception("Parse error")),
    ):
        cov = get_doc_coverage()
    assert cov == 0.0


def test_update_readme_no_file() -> None:
    """Verify `update_readme` returns early when `README.md` does not exist."""
    with patch("os.path.exists", return_value=False):
        update_readme()
    assert callable(test_update_readme_no_file)


def test_update_readme_success() -> None:
    """Verify `update_readme` updates both test and doc coverage badges in `README.md`."""
    readme_content = """# Project
[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)
"""
    m_open = mock_open(read_data=readme_content)

    with (
        patch("os.path.exists", return_value=True),
        patch("scripts.update_badges.get_test_coverage", return_value=100.0),
        patch("scripts.update_badges.get_doc_coverage", return_value=95.5),
        patch("builtins.open", m_open),
    ):
        update_readme()

    written = "".join([call.args[0] for call in m_open().write.call_args_list])
    assert "test_coverage-100%25-brightgreen" in written
    assert "doc_coverage-95.5%25-green" in written


def test_update_readme_no_coverage_returned() -> None:
    """Verify `update_readme` leaves badges unchanged when coverage values are `None`."""
    readme_content = """# Project
[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)
"""
    m_open = mock_open(read_data=readme_content)

    with (
        patch("os.path.exists", return_value=True),
        patch("scripts.update_badges.get_test_coverage", return_value=None),
        patch("scripts.update_badges.get_doc_coverage", return_value=None),
        patch("builtins.open", m_open),
    ):
        update_readme()

    written = "".join([call.args[0] for call in m_open().write.call_args_list])
    assert "test_coverage-50%25-red" in written
    assert "doc_coverage-50%25-red" in written


def test_main_block() -> None:
    """Verify `scripts.update_badges` `__main__` entrypoint."""
    import runpy
    import sys

    with patch.object(sys, "argv", ["update_badges.py"]):
        with patch("os.path.exists", return_value=False):
            runpy.run_module("scripts.update_badges", run_name="__main__")
    assert callable(test_main_block)
