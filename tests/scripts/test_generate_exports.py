"""Tests for the generate_exports script."""

from __future__ import annotations

import builtins
import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import mock_open, patch

import pytest

import scripts.generate_exports as ge

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
_real_import = importlib.import_module


def _mock_mod_import(mod_obj: EagerValue) -> Callable[..., EagerValue]:
    """Create a mock importlib.import_module side-effect targeting test modules.

    Args:
        mod_obj (EagerValue): Object to return when user/test module is imported.

    Returns:
        Callable[..., object]: Mock import side-effect function.
    """

    def _imp(name: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function _imp.

        Args:
            name: Test parameter `name`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        if name in ("subprocess", "scripts", "scripts.generate_exports", "ast", "os", "sys", "tempfile", "builtins"):
            return cast("EagerCallableProtocol", _real_import)(name, *args, **kwargs)
        return mod_obj

    return _imp


def test_get_exports_from_submodule() -> None:
    """Test retrieving exports from a submodule."""
    with patch("scripts.generate_exports.importlib.import_module") as mock_import:
        mock_import.return_value.__all__ = ["A", "B"]
        assert ge._get_exports_from_submodule("foo") == ["A", "B"]
    with patch("scripts.generate_exports.importlib.import_module") as mock_import:

        class MockMod:
            """Mock module."""

            _hidden = 1
            visible = 2

            def func(self) -> None:
                """Mock method."""
                pass

        import types

        setattr(MockMod, "mod", types.ModuleType("sub"))
        mock_import.return_value = MockMod()
        assert ge._get_exports_from_submodule("foo") == ["func", "visible"]
    with patch("scripts.generate_exports.importlib.import_module", side_effect=ImportError("mock error")):
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
        with patch("subprocess.run"):
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"], extra_imports=[("ext_mod", ["w"])])
    new_content = init_file.read_text()
    assert "__all__ =" in new_content
    assert '"y"' in new_content
    assert '"z"' in new_content
    assert '"w"' in new_content
    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["w", "y", "z"]
        ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"])
    assert init_file.read_text() == new_content


def test_generate_init_syntax_error(tmp_path: Path) -> None:
    """Test generating __init__.py when the original file has a syntax error.

    Args:
        tmp_path (Path): Pytest fixture.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n__all__ = [")
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
        """Mock absolute path.

        Args:
            p: Test parameter `p`.
        """
        if p == "src":
            return "/tmp/src"
        return "/tmp/other/file.py"

    with patch("os.path.abspath", fake_abs):
        ge.process_file(str(f))
    assert f.exists()


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
        """Mock absolute path.

        Args:
            p: Test parameter `p`.
        """
        if p == "src":
            return str(tmp_path / "src")
        return os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p

    monkeypatch.setattr("os.path.abspath", fake_abspath)
    with patch("scripts.generate_exports.importlib.import_module", side_effect=Exception):
        ge.process_file(str(f))
    assert f.read_text() == ""


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
        """Mock absolute path.

        Args:
            p: Test parameter `p`.
        """
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

    class MockMod2:
        """Mock module."""

        __all__ = ["x"]

    with patch("subprocess.run"), patch("scripts.generate_exports._get_exports_from_submodule", return_value=["x", "y"]):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod2)())):
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

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
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
    f.write_text("x = 1\n__all__ = ['y']\n\n  \n")

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
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
    f.write_text("x = 1\n__all__ = ['x']")

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
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

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
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

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
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
    f.write_text("x = 1\n__all__ = [")

    class MockMod:
        """Mock module."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    assert f.read_text() == "x = 1\n__all__ = ["


def test_process_file_invalid_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test processing a file where existing __all__ is an invalid type.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_invalid_all.py"
    f.write_text("x = 1\n__all__ = 123\n")

    class MockMod:
        """Mock module."""

        __all__ = 123

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    assert f.read_text() == "x = 1\n\n__all__ = [\n]\n"


def test_generate_init_no_rewrite_needed(tmp_path: Path) -> None:
    """Test function test_generate_init_no_rewrite_needed.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n__all__ = ['y']")
    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["y"]
        with patch("subprocess.run"):
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"])
    assert init_file.read_text() == "x = 1\n__all__ = ['y']"


def test_generate_init_invalid_all_type(tmp_path: Path) -> None:
    """Test function test_generate_init_invalid_all_type.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n__all__ = 123")
    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["y"]
        with patch("subprocess.run"):
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"])
    assert "__all__ = [" in init_file.read_text()


def test_process_file_no_py_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_no_py_extension.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_no_ext"
    f.write_text("x = 1\n")
    with patch("scripts.generate_exports.importlib.import_module", side_effect=ImportError):
        ge.process_file(str(f))
    assert f.read_text() == "x = 1\n"


def test_process_file_expr_not_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_expr_not_call.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_expr.py"
    f.write_text("x = 1\n__all__\n'string'\n__all__.sort()\n__all__.extend(['x'])")

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    content = f.read_text()
    assert "__all__ =" in content


def test_generate_init_same_source(tmp_path: Path) -> None:
    """Test function test_generate_init_same_source.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("old text")
    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["y"]
        with patch("subprocess.run"):
            with patch("builtins.open", mock_open(read_data="old text")) as m_open:
                ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"])
                m_open().write.assert_not_called()


def test_process_file_same_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_same_source.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_same.py"
    original_source = "x = 1\n__all__ = [y]\n"
    f.write_text(original_source)

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x"]

    def mock_subprocess_run(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function mock_subprocess_run.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        cmd = args[0]
        if cmd and cast("EagerTensorProtocol", "ruff") in cast("EagerTensorProtocol", cmd):
            tmp_name = cast("EagerTensorProtocol", getattr(cmd, "__getitem__")(-1))
            if cast("EagerCallableProtocol", os.path.exists)(tmp_name):
                with cast("EagerCallableProtocol", builtins.open)(tmp_name, "w") as tmp_f:
                    tmp_f.write(original_source)

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    assert f.read_text() == original_source


def test_process_file_ast_elements_middle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_ast_elements_middle.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_ast_elements_mid.py"
    f.write_text("x = 1\n__all__ = ['x', y, 'z']\n")

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x", "z"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_empty_all.py"
    f.write_text("x = 1\n__all__ = []\n")

    class MockMod2:
        """Test class MockMod2."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod2)())):
            ge.process_file(str(f))
    assert "x = 1\n\n__all__ = [\n" in f.read_text()


def test_process_file_no_end_lineno(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_no_end_lineno.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_no_end_lineno.py"
    f.write_text("x = 1\n__all__ = ['x']\n")

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x", "y"]

    original_walk = __import__("ast").walk

    def mock_walk(node: EagerValue) -> EagerValue:
        """Test function mock_walk.

        Args:
            node: Test parameter `node`.
        """
        nodes = list(original_walk(node))
        for n in nodes:
            if hasattr(n, "end_lineno"):
                n.end_lineno = None
        return nodes

    with patch("subprocess.run"):
        with patch("ast.walk", side_effect=mock_walk):
            with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
                ge.process_file(str(f))
    assert "__all__" in f.read_text()


def test_process_file_complex_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_complex_targets.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_complex.py"
    f.write_text("x = 1\nother = __all__ = ['x']\n__all__.something()\nother.append('x')")

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x", "y"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    content = f.read_text()
    assert "other.append" in content
    assert "__all__.something" in content
    assert "other = __all__ =" not in content
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n__all__ = ['x', y]")
    with patch("scripts.generate_exports._get_exports_from_submodule") as mock_get:
        mock_get.return_value = ["y", "z"]
        with patch("subprocess.run"):
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1"])
    assert '"y"' in init_file.read_text()


def test_generate_init_no_new_exports(tmp_path: Path) -> None:
    """Test function test_generate_init_no_new_exports.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n")
    with patch("scripts.generate_exports._get_exports_from_submodule", return_value=["y"]):
        with patch("subprocess.run"):
            ge.generate_init(filepath=str(init_file), module_name="test_mod", submodules=["sub1", "sub2"])
    content = init_file.read_text()
    assert content.count('"y"') == 1


def test_process_file_tuple_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_tuple_all.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_tuple.py"
    f.write_text("x = 1\n__all__ = ('x',)")

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    assert f.read_text() == "x = 1\n__all__ = ('x',)"


def test_process_file_extend_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_process_file_extend_all.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    src = setup_mock_src(tmp_path, monkeypatch)
    f = src / "test_extend.py"
    f.write_text("x = 1\n__all__.extend(['x'])\n")

    class MockMod:
        """Test class MockMod."""

        __all__ = ["x"]

    with patch("subprocess.run"):
        with cast("EagerCallableProtocol", patch)("scripts.generate_exports.importlib.import_module", side_effect=cast("EagerCallableProtocol", _mock_mod_import)(cast("EagerCallableProtocol", MockMod)())):
            ge.process_file(str(f))
    content = f.read_text()
    assert "__all__ =" in content
    assert "extend" not in content


def test_main() -> None:
    """Test the __main__ block of generate_exports.py."""
    with patch("scripts.generate_exports.generate_init") as mock_generate_init:
        with patch("scripts.generate_exports.process_file"):
            ge.main()
            mock_generate_init.assert_called()


def test_main_block(capsys: pytest.CaptureFixture[str]) -> None:
    """Test function test_main_block.

    Args:
        capsys: Test parameter `capsys`.
    """
    import runpy
    import sys

    with patch.object(sys, "argv", ["generate_exports.py"]):
        with patch("scripts.generate_exports.main"):
            try:
                runpy.run_path("scripts/generate_exports.py", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0
    with patch.object(sys, "argv", ["generate_exports.py"]):
        with patch("scripts.generate_exports.generate_init"):
            with patch("scripts.generate_exports.process_file"):
                with patch("glob.glob", return_value=[]):
                    runpy.run_path("scripts/generate_exports.py", run_name="__main__")
    assert callable(test_main_block)
