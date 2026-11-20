"""Build script to dynamically regenerate __all__ literal string lists."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
import types
import typing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.protocols import (
    BackendOperand,
    adapt_backend_module,
)


@typing.runtime_checkable
class _HasAllAttrProtocol(typing.Protocol):
    """Protocol for modules or mocks exposing an `__all__` attribute."""

    __all__: list[str] | tuple[str, ...] | set[str] | AttributeValue


def _has_nonempty_all(mod: AttributeValue | types.ModuleType) -> bool:
    """Return True if `mod` defines a non-empty `__all__` member.

    Args:
        mod (AttributeValue | types.ModuleType): Target module or object.

    Returns:
        bool: True if `__all__` exists and is truthy.
    """
    if isinstance(mod, _HasAllAttrProtocol):
        return bool(mod.__all__)
    return False


def _lookup_all_exports(mod: AttributeValue | types.ModuleType) -> list[str] | None:
    """Look up `__all__` on `mod` via structural protocol check if it is a sequence or set.

    Args:
        mod (AttributeValue | types.ModuleType): Target module or object.

    Returns:
        list[str] | None: Sorted list of exported names if `__all__` is a sequence/set, `[]` if defined as another
            type, or `None` if absent.
    """
    if isinstance(mod, _HasAllAttrProtocol):
        v = mod.__all__
        if isinstance(v, (list, tuple, set)):
            return sorted({str(x) for x in v})
        return [] if v is not None else None
    return None


def _ast_defines_non_module_symbol(filepath: str | None, name: str) -> bool:
    """Return True if `filepath` explicitly defines or imports `name` as a non-submodule symbol.

    Args:
        filepath (str | None): Module source file path.
        name (str): Candidate symbol identifier.

    Returns:
        bool: True when `name` is explicitly bound in the module AST as a function, class, variable, or symbol import.
    """
    if not filepath or not os.path.isfile(filepath):
        return False
    try:
        with open(filepath, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
        is_assign = isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        is_ann = isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name
        is_imp = (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and any((alias.asname or alias.name) == name for alias in node.names)
        )
        if is_def or is_assign or is_ann or is_imp:
            return True
    return False


def _is_submodule_member(mod: BackendOperand | types.ModuleType, name: str) -> bool:
    """Return True if member `name` on `mod` is a `types.ModuleType`.

    Args:
        mod (BackendOperand | types.ModuleType): Target module or object.
        name (str): Member name to inspect.

    Returns:
        bool: True if the member is a module.
    """
    if isinstance(mod, types.ModuleType) and f"{mod.__name__}.{name}" in sys.modules:
        if not _ast_defines_non_module_symbol(mod.__file__, name):
            return True
    found, raw_val = adapt_backend_module(mod).resolve_symbol(name)
    return bool(found and isinstance(raw_val, types.ModuleType))


def _get_exports_from_submodule(modname: str) -> list[str]:
    """Retrieve the exported symbols from a submodule.

    Args:
        modname (str): The fully qualified name of the module to import.

    Returns:
        list[str]: A sorted list of the module's exported symbol names.
    """
    try:
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        mod: types.ModuleType = BackendRegistry.load_scoped_module(modname)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        print(f"Failed to import {modname}: {e}")
        return []

    all_exports = _lookup_all_exports(mod)
    if all_exports is not None and all_exports:
        return all_exports
    return sorted(
        {n for n in dir(mod) if not n.startswith("_") and n != "annotations" and not _is_submodule_member(mod, n)}
    )


def _append_import_lines(
    modname: str,
    exports: list[str],
    imported_symbols: set[str],
    import_lines: list[str],
    all_exports: list[str],
) -> None:
    """Append import statements for new exports to the given lists.

    Args:
        modname (str): The module name to import from.
        exports (list[str]): The list of symbols to import.
        imported_symbols (set[str]): The set of symbols that have already been imported.
        import_lines (list[str]): The list of formatted import statements to append to.
        all_exports (list[str]): The list of all exported symbols to append to.

    Returns:
        None
    """
    unique_exports: list[str] = [e for e in sorted(set(exports)) if e not in imported_symbols]
    if unique_exports:
        import_lines.append(f"from {modname} import (")
        for e in unique_exports:
            import_lines.append(f"    {e},")
        import_lines.append(")")
        imported_symbols.update(unique_exports)
        all_exports.extend(unique_exports)


def _extract_existing_all(source: str) -> list[str] | None:
    """Extract existing __all__ string literals from Python source.

    Args:
        source (str): Python source code.

    Returns:
        Optional[list[str]]: List of exported names if found, else None.
    """
    try:
        tree: ast.Module = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    return [
                        str(elt.value)
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    return None


def _format_and_write_if_changed(filepath: str, old_source: str, new_source: str) -> None:
    """Format new_source with ruff and write to filepath if different from old_source.

    Args:
        filepath (str): Target file path.
        old_source (str): Original source text.
        new_source (str): New source text to format and write.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(new_source)
        tmp_name: str = tmp.name

    try:
        subprocess.run(
            ["ruff", "check", "--fix", "--unsafe-fixes", "--config", "pyproject.toml", tmp_name],
            capture_output=True,
            check=False,
        )
        subprocess.run(["ruff", "format", "--config", "pyproject.toml", tmp_name], capture_output=True, check=False)
        with open(tmp_name) as f:
            new_source_formatted: str = f.read()
    finally:
        os.remove(tmp_name)

    if old_source != new_source_formatted:
        with open(filepath, "w") as f:
            f.write(new_source_formatted)
        print(f"Updated {filepath}")


def generate_init(
    filepath: str,
    module_name: str,
    submodules: list[str],
    extra_imports: list[tuple[str, list[str]]] | None = None,
) -> None:
    """Generate the __init__.py file with explicit imports and __all__ list.

    Args:
        filepath (str): The path to the __init__.py file to write.
        module_name (str): The base name of the module.
        submodules (list[str]): A list of submodules to import from.
        extra_imports (Optional[list[tuple[str, list[str]]]]): A list of tuples containing an external module name and
            a list of symbols to import.

    Returns:
        None
    """
    all_exports: list[str] = []
    import_lines: list[str] = []
    imported_symbols: set[str] = set()

    for submod in submodules:
        modname: str = f"{module_name}.{submod}"
        exports: list[str] = _get_exports_from_submodule(modname)
        _append_import_lines(modname, exports, imported_symbols, import_lines, all_exports)

    if extra_imports:
        for extra_modname, extra_exports_list in extra_imports:
            _append_import_lines(extra_modname, extra_exports_list, imported_symbols, import_lines, all_exports)

    all_exports = sorted(list(set(all_exports)))

    with open(filepath) as f:
        old_source: str = f.read()

    existing_all = _extract_existing_all(old_source)
    if existing_all is not None and sorted(list(set(existing_all))) == all_exports:
        return

    new_source: str = (
        (
            f'# pylint: disable=too-many-lines\n"""Public package exports for `{module_name}`."""\n\nfrom '
            "__future__ import annotations\n\n"
        )
        + "\n".join(import_lines)
        + "\n\n"
    )
    new_source += "# pylint: disable=duplicate-code\n"
    new_source += "__all__ = [\n"
    for e in all_exports:
        new_source += f'    "{e}",\n'
    new_source += "]\n"

    _format_and_write_if_changed(filepath, old_source, new_source)
    sys.modules.pop(module_name, None)


def _compute_module_exports(mod: types.ModuleType, source_for_magic: str) -> list[str]:
    """Compute sorted list of exported symbol names for a module.

    Args:
        mod (types.ModuleType): Imported module object.
        source_for_magic (str): Source text to inspect for magic comments.

    Returns:
        list[str]: Sorted list of exported symbol names.
    """
    magic_from: re.Match[str] | None = re.search(r"#\s*generate_exports_from:\s*(.+)", source_for_magic)
    magic_auto: re.Match[str] | None = re.search(r"#\s*auto-generate-all", source_for_magic)
    exclude_set: set[str] = {
        m.strip()
        for line in re.findall(r"#\s*exclude_exports:\s*(.+)", source_for_magic)
        for m in line.split(",")
        if m.strip()
    }

    if magic_from:
        raw_exports: list[str] = []
        for sm in (m.strip() for m in magic_from.group(1).split(",")):
            raw_exports.extend(_get_exports_from_submodule(sm))
        return sorted({e for e in raw_exports if e not in exclude_set})
    if magic_auto:
        return sorted(
            {
                n
                for n in dir(mod)
                if not n.startswith("_")
                and n != "annotations"
                and not _is_submodule_member(mod, n)
                and n not in exclude_set
            }
        )

    exports_attr = _lookup_all_exports(mod)
    if exports_attr is None:
        return sorted({n for n in dir(mod) if not n.startswith("_") and n != "annotations"})
    return exports_attr


def _find_all_lines_to_remove(tree: ast.Module) -> set[int]:
    """Identify 1-based line numbers of __all__ assignments and mutations in an AST.

    Args:
        tree (ast.Module): Parsed module AST.

    Returns:
        set[int]: Line numbers to remove.
    """
    lines_to_remove: set[int] = set()
    for node in ast.walk(tree):
        is_assign = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        )
        is_call = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "__all__"
            and node.value.func.attr in ("extend", "append")
        )
        if (is_assign or is_call) and isinstance(node, ast.stmt):
            lineno = node.lineno
            end_lineno = node.end_lineno
            if isinstance(end_lineno, int):
                for line_num in range(lineno, end_lineno + 1):
                    lines_to_remove.add(line_num)
    return lines_to_remove


def process_file(filepath: str) -> None:
    """Process a single file to update its __all__ exports.

    Args:
        filepath (str): The path to the Python file to process.

    Returns:
        None
    """
    abs_path: str = os.path.abspath(filepath)
    src_dir: str = os.path.abspath("src")
    if not abs_path.startswith(src_dir):
        return

    modname: str = os.path.relpath(abs_path, src_dir).replace(os.path.sep, ".")
    if modname.endswith(".py"):
        modname = modname[:-3]
    if modname.endswith(".__init__"):
        modname = modname[:-9]

    try:
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        mod: types.ModuleType = BackendRegistry.load_scoped_module(modname)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        print(f"Skipping {filepath} due to import error: {e}")
        return

    with open(filepath) as f:
        source: str = f.read()

    exports = _compute_module_exports(mod, source)
    existing_all = _extract_existing_all(source)
    if existing_all is not None and sorted(list(set(existing_all))) == exports:
        return

    try:
        tree: ast.Module = ast.parse(source)
    except SyntaxError:
        return

    lines_to_remove = _find_all_lines_to_remove(tree)
    if not lines_to_remove and not _has_nonempty_all(mod):
        return

    new_lines: list[str] = [line for i, line in enumerate(source.split("\n"), 1) if i not in lines_to_remove]
    while new_lines and new_lines[-1].strip() == "":
        new_lines.pop()

    new_source: str = "\n".join(new_lines) + "\n\n__all__ = [\n"
    for e_exp in exports:
        new_source += f'    "{e_exp}",\n'
    new_source += "]\n"

    _format_and_write_if_changed(filepath, source, new_source)
    sys.modules.pop(modname, None)


def main() -> None:
    """Main execution block.

    Returns:
        None
    """
    vision_subs: list[str] = [
        "affine",
        "bbox",
        "color",
        "filtering",
        "interpolation",
        "mixing",
        "transforms",
        "api",
    ]
    # First, generate vision exports specifically
    generate_init(
        "src/ml_switcheroo_compiler/ops/vision/__init__.py",
        "ml_switcheroo_compiler.ops.vision",
        vision_subs,
    )

    # Then process all files to convert __all__ to literal strings
    files_to_check: list[str] = []
    for root, _, files in os.walk("src/ml_switcheroo_compiler"):
        for file in files:
            if file.endswith(".py"):
                fpath: str = os.path.join(root, file)
                if fpath != "src/ml_switcheroo_compiler/ops/vision/__init__.py":
                    files_to_check.append(fpath)

    for f in files_to_check:
        with open(f) as file_obj:
            content: str = file_obj.read()
        if "__all__" in content:
            process_file(f)

    print("Successfully normalized __all__ exports across the codebase.")

    ops_subs: list[str] = [
        "audio",
        "base",
        "foreign_call",
        "control_flow",
        "creation",
        "io",
        "shape",
        "text",
        "signal",
        "sparse",
        "ragged",
        "tensor_array",
        "state",
        "device",
        "raw_ops",
        "creation.api",
        "binary",
        "unary",
        "linalg",
        "reductions",
        "random_ops",
        "nn",
        "normalization",
        "vision",
        "image",
        "stats",
    ]
    generate_init(
        "src/ml_switcheroo_compiler/ops/__init__.py",
        "ml_switcheroo_compiler.ops",
        ops_subs,
    )


if __name__ == "__main__":
    main()
