"""Tests for the generate_exports script."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.generate_exports as ge


def test_get_exports_from_submodule() -> None:
    """Test retrieving exports from a submodule."""
    # Test valid import
    with patch("importlib.import_module") as mock_import:
        mock_import.return_value.__all__ = ["A", "B"]
        assert ge._get_exports_from_submodule("foo") == ["A", "B"]

    # Test import without __all__
    with patch("importlib.import_module") as mock_import:

        class MockMod:
            """Mock module."""

            _hidden = 1
            visible = 2

            def func(self) -> None:
                """Mock method."""
                pass

        import types

        MockMod.mod = types.ModuleType("sub")
        mock_import.return_value = MockMod()
        assert ge._get_exports_from_submodule("foo") == ["func", "visible"]

    # Test import error
    with patch("importlib.import_module", side_effect=ImportError("mock error")):
        assert ge._get_exports_from_submodule("foo") == []


def test_generate_init(tmp_path: Path) -> None:
    """Test generating __init__.py with imports and __all__.

    Args:
        tmp_path (Path): Pytest fixture.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n__all__ = ['x']")

    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["y", "z"]

        with patch("subprocess.run"):  # don't run ruff in test
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"], extra_imports=[("ext_mod", ["w"])])

    new_content = init_file.read_text()
    assert "__all__ =" in new_content
    assert '"y"' in new_content
    assert '"z"' in new_content
    assert '"w"' in new_content

    # Test when they match (should return early)
    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["w", "y", "z"]
        ge.generate_init(
            filepath=str(init_file),
            module_name="test_mod",
            submodules=["sub1"],
        )
    assert init_file.read_text() == new_content


def test_generate_init_syntax_error(tmp_path: Path) -> None:
    """Test generating __init__.py when the original file has a syntax error.

    Args:
        tmp_path (Path): Pytest fixture.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n__all__ = [")  # syntax error

    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["y", "z"]

        with patch("subprocess.run"):
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"])
    assert '"y"' in init_file.read_text()


def test_process_file_no_src_dir(tmp_path: Path) -> None:
    """Test processing a file that is not in the src directory.

    Args:
        tmp_path (Path): Pytest fixture.
    """
    f = tmp_path / "some_file.py"
    f.write_text("")

    def fake_abs(p: str) -> str:
        """Mock absolute path."""
        if p == "src":
            return "/tmp/src"
        return "/tmp/other/file.py"

    with patch("os.path.abspath", fake_abs):
        ge.process_file(str(f))


def test_process_file_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file that fails to import.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    f = tmp_path / "src" / "test.py"
    f.parent.mkdir()
    f.write_text("")

    def fake_abspath(p: str) -> str:
        """Mock absolute path."""
        if p == "src":
            return str(tmp_path / "src")
        return os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p

    monkeypatch.setattr("os.path.abspath", fake_abspath)

    with patch("importlib.import_module", side_effect=Exception):
        ge.process_file(str(f))


def setup_mock_src(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Setup a mock src directory for testing.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        Path: The mock src directory.
    """
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)

    def fake_abspath(p: str) -> str:
        """Mock absolute path."""
        if p == "src":
            return str(src)
        if not os.path.isabs(p):
            return str(tmp_path / p)
        return p

    monkeypatch.setattr("os.path.abspath", fake_abspath)
    return src


def test_process_file_magic_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file with 'generate_exports_from' magic comment.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test.py"
    f.write_text("# generate_exports_from: sub1\n# exclude_exports: y\nx = 1\n__all__ = ['w']\n\n\n")

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("scripts.generate_exports._get_exports_from_submodule", return_value=["x", "y"]):
            with patch("subprocess.run"):
                ge.process_file(str(f))

    content = f.read_text()
    assert "__all__ =" in content
    assert '"x"' in content
    assert '"y"' not in content


def test_process_file_magic_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file with 'auto-generate-all' magic comment.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test2.py"
    f.write_text("# auto-generate-all\nx = 1\ny = 2\n_z = 3\n__all__ = ['w']")

    class MockMod:
        """Mock module."""

        x = 1
        y = 2
        _z = 3

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            ge.process_file(str(f))

    content = f.read_text()
    assert '"x"' in content
    assert '"y"' in content
    assert '"_z"' not in content


def test_process_file_no_magic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file with no magic comments but existing __all__.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test3.__init__.py"
    f.write_text("x = 1\n__all__ = ['y']\n\n  \n")  # different existing, trailing space for coverage

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            ge.process_file(str(f))

    content = f.read_text()
    assert '"x"' in content
    assert '"y"' not in content


def test_process_file_existing_all_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file where existing __all__ exactly matches.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_match.py"
    f.write_text("x = 1\n__all__ = ['x']")  # matches exactly

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            ge.process_file(str(f))

    assert f.read_text() == "x = 1\n__all__ = ['x']"


def test_process_file_no_magic_no_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file with no magic comments and no existing __all__.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test4.py"
    f.write_text("x = 1\n")

    class MockMod:
        """Mock module."""

        __all__ = None
        x = 1

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            ge.process_file(str(f))

    assert f.read_text() == "x = 1\n"


def test_process_file_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file that appends to __all__.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test6.py"
    f.write_text("x = 1\n__all__.append('x')")

    class MockMod:
        """Mock module."""

        __all__ = ["x"]
        x = 1

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            ge.process_file(str(f))

    content = f.read_text()
    assert "__all__ =" in content
    assert "__all__.append" not in content


def test_process_file_syntax_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file with a syntax error.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test5.py"
    f.write_text("x = 1\n__all__ = [")  # Invalid syntax

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            # Ensure it hits the return line
            ge.process_file(str(f))

    # Content shouldn't change due to syntax error
    assert f.read_text() == "x = 1\n__all__ = ["


def test_process_file_invalid_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file where existing __all__ is an invalid type.

    Args:
        tmp_path (Path): Pytest fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_invalid_all.py"
    f.write_text("x = 1\n")

    class MockMod:
        """Mock module."""

        __all__ = 123  # Not a list, tuple, or set

    with patch("importlib.import_module", return_value=MockMod()):
        with patch("subprocess.run"):
            ge.process_file(str(f))

    assert f.read_text() == "x = 1\n\n__all__ = [\n]\n"


def test_main() -> None:
    """Test the __main__ block of generate_exports.py."""
    with patch("scripts.generate_exports.generate_init") as mock_generate_init:
        with patch("scripts.generate_exports.process_file") as mock_process_file:
            ge.main()
            mock_generate_init.assert_called()
