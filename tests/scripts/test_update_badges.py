import json
import subprocess
from unittest.mock import mock_open, patch

from scripts.update_badges import (
    format_cov,
    get_color,
    get_doc_coverage,
    get_test_coverage,
    update_readme,
)


def test_get_color():
    assert get_color(100) == "brightgreen"
    assert get_color(95) == "green"
    assert get_color(85) == "yellowgreen"
    assert get_color(75) == "yellow"
    assert get_color(65) == "orange"
    assert get_color(50) == "red"


def test_format_cov():
    assert format_cov(100.0) == "100"
    assert format_cov(95.5) == "95.5"


def test_get_test_coverage_success():
    mock_data = {"totals": {"percent_covered": 85.5}}
    m_open = mock_open(read_data=json.dumps(mock_data))

    with patch("subprocess.run") as mock_run, patch("builtins.open", m_open):
        cov = get_test_coverage()

    mock_run.assert_called_once()
    assert cov == 85.5


def test_get_test_coverage_failure():
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
        cov = get_test_coverage()
    assert cov is None


def test_get_doc_coverage_no_dir():
    with patch("os.path.exists", return_value=False):
        assert get_doc_coverage() == 0.0


def test_get_doc_coverage_no_files():
    with patch("os.path.exists", return_value=True), patch("os.walk", return_value=[]):
        assert get_doc_coverage() == 0.0


def test_get_doc_coverage_success():
    mock_walk = [("src/ml_switcheroo_compiler", [], ["__init__.py", "core.py", "bad.py", "not_python.txt"])]

    # core.py has docstrings
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
    # bad.py has no docstrings and some private members
    bad_content = """class _Private:
    def _priv_method(self):
        pass
    async def _priv_async(self):
        pass

def _priv_func():
    pass

async def _module_priv_async():
    pass

class NoDoc:
    def no_doc_method(self):
        pass
    async def no_doc_async(self):
        pass
    async def _inner_priv_async(self):
        pass

def no_doc_func():
    pass
"""

    def mock_open_file(path, *args, **kwargs):
        if "core.py" in path:
            return mock_open(read_data=core_content)()
        elif "bad.py" in path:
            return mock_open(read_data=bad_content)()
        raise Exception("File not found")

    with patch("os.path.exists", return_value=True), patch("os.walk", return_value=mock_walk), patch("builtins.open", side_effect=mock_open_file):
        cov = get_doc_coverage()

    assert cov == 50.0


def test_get_doc_coverage_exception_in_file():
    mock_walk = [("src/ml_switcheroo_compiler", [], ["fail.py"])]
    with patch("os.path.exists", return_value=True), patch("os.walk", return_value=mock_walk), patch("builtins.open", mock_open()), patch("ast.parse", side_effect=Exception("Parse error")):
        cov = get_doc_coverage()
    assert cov == 0.0


def test_update_readme_no_file():
    with patch("os.path.exists", return_value=False):
        update_readme()


def test_update_readme_success():
    readme_content = """# Project
[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)
"""
    m_open = mock_open(read_data=readme_content)

    with patch("os.path.exists", return_value=True), patch("scripts.update_badges.get_test_coverage", return_value=100.0), patch("scripts.update_badges.get_doc_coverage", return_value=95.5), patch("builtins.open", m_open):
        update_readme()

    written = "".join([call.args[0] for call in m_open().write.call_args_list])
    assert "test_coverage-100%25-brightgreen" in written
    assert "doc_coverage-95.5%25-green" in written


def test_update_readme_no_coverage_returned():
    readme_content = """# Project
[![Test Coverage](https://img.shields.io/badge/test_coverage-50%25-red.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-50%25-red.svg)](#)
"""
    m_open = mock_open(read_data=readme_content)

    with patch("os.path.exists", return_value=True), patch("scripts.update_badges.get_test_coverage", return_value=None), patch("scripts.update_badges.get_doc_coverage", return_value=None), patch("builtins.open", m_open):
        update_readme()

    written = "".join([call.args[0] for call in m_open().write.call_args_list])
    assert "test_coverage-50%25-red" in written
    assert "doc_coverage-50%25-red" in written
