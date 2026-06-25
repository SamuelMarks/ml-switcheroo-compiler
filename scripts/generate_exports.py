"""Build script to dynamically regenerate __all__ literal string lists."""

import ast
import importlib
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def _get_exports_from_submodule(modname: str) -> list[str]:
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        print(f"Failed to import {modname}: {e}")
        return []

    if hasattr(mod, "__all__"):
        return sorted(list(set(mod.__all__)))
    return sorted(list(set([n for n in dir(mod) if not n.startswith("_")])))


def _append_import_lines(
    modname: str,
    exports: list[str],
    imported_symbols: set[str],
    import_lines: list[str],
    all_exports: list[str],
) -> None:
    unique_exports = [e for e in sorted(set(exports)) if e not in imported_symbols]
    if unique_exports:
        import_lines.append(f"from {modname} import (")
        for e in unique_exports:
            import_lines.append(f"    {e},")
        import_lines.append(")")
        imported_symbols.update(unique_exports)
        all_exports.extend(unique_exports)


def generate_init(
    filepath: str,
    module_name: str,
    submodules: list[str],
    extra_imports: Optional[list[tuple[str, list[str]]]] = None,
) -> None:
    """Generate the __init__.py file with explicit imports and __all__ list.

    Args:
        filepath: The path to the __init__.py file to overwrite.
        module_name: The base package name.
        submodules: A list of submodule names.
        extra_imports: A list of tuples containing an import path and symbols.
    """
    all_exports: list[str] = []
    import_lines: list[str] = []
    imported_symbols: set[str] = set()

    for submod in submodules:
        modname = f"{module_name}.{submod}"
        exports = _get_exports_from_submodule(modname)
        _append_import_lines(modname, exports, imported_symbols, import_lines, all_exports)

    if extra_imports:
        for modname, exports in extra_imports:
            _append_import_lines(modname, exports, imported_symbols, import_lines, all_exports)

    all_exports = sorted(list(set(all_exports)))

    with open(filepath) as f:
        old_source = f.read()

    new_source = (
        f'# pylint: disable=too-many-lines\n"""Auto-generated {module_name} module exports."""\n\n'
    )
    new_source += "\n".join(import_lines) + "\n\n"
    new_source += "__all__ = [\n"
    for e in all_exports:
        new_source += f'    "{e}",\n'
    new_source += "]\n"

    try:
        if ast.dump(ast.parse(old_source)) != ast.dump(ast.parse(new_source)):
            with open(filepath, "w") as f:
                f.write(new_source)
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Failed to parse AST for {filepath}: {e}")


def process_file(filepath: str) -> None:  # noqa: C901, PLR0912, PLR0915
    """Process a single file to update its __all__ exports."""
    abs_path = os.path.abspath(filepath)
    src_dir = os.path.abspath("src")
    if not abs_path.startswith(src_dir):
        return

    rel_path = os.path.relpath(abs_path, src_dir)
    modname = rel_path.replace(os.path.sep, ".")
    if modname.endswith(".py"):
        modname = modname[:-3]
    if modname.endswith(".__init__"):
        modname = modname[:-9]

    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        print(f"Skipping {filepath} due to import error: {e}")
        return

    exports = getattr(mod, "__all__", None)
    if exports is None:
        exports = [n for n in dir(mod) if not n.startswith("_")]

    exports = sorted(list(set(exports)))

    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    lines_to_remove = set()
    for node in ast.walk(tree):
        is_target = False
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    is_target = True
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "__all__"
                ):
                    if func.attr in ("extend", "append"):
                        is_target = True

        if is_target:
            for line_num in range(node.lineno, node.end_lineno + 1):
                lines_to_remove.add(line_num)

    if not lines_to_remove and not getattr(mod, "__all__", None):
        return

    old_lines = source.split("\n")
    new_lines = []
    for i, line in enumerate(old_lines, 1):
        if i not in lines_to_remove:
            new_lines.append(line)

    while new_lines and new_lines[-1].strip() == "":
        new_lines.pop()

    new_source = "\n".join(new_lines) + "\n\n__all__ = [\n"
    for e in exports:
        new_source += f'    "{e}",\n'
    new_source += "]\n"

    try:
        old_dump = ast.dump(ast.parse(source))
        new_dump = ast.dump(ast.parse(new_source))
    except Exception as e:
        print(f"Error parsing new source for {filepath}: {e}")
        return

    if old_dump != new_dump:
        with open(filepath, "w") as f:
            f.write(new_source)
        print(f"Updated {filepath}")


if __name__ == "__main__":
    vision_subs = [
        "affine",
        "bbox",
        "color",
        "filtering",
        "interpolation",
        "mixing",
        "transforms",
    ]
    # First, generate vision exports specifically
    generate_init(
        "src/ml_switcheroo_compiler/ops/vision/__init__.py",
        "ml_switcheroo_compiler.ops.vision",
        vision_subs,
    )

    # Then process all files to convert __all__ to literal strings
    files_to_check = []
    for root, _, files in os.walk("src/ml_switcheroo_compiler"):
        for file in files:
            if file.endswith(".py"):
                fpath = os.path.join(root, file)
                if fpath != "src/ml_switcheroo_compiler/ops/vision/__init__.py":
                    files_to_check.append(fpath)

    for f in files_to_check:
        with open(f) as file_obj:
            content = file_obj.read()
        if "__all__" in content:
            process_file(f)

    print("Successfully normalized __all__ exports across the codebase.")
