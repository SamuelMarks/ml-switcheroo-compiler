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


def _get_existing_all(filepath: str) -> Optional[list[str]]:
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            return [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
    except Exception:
        pass
    return None


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
    existing_all = _get_existing_all(filepath)

    if existing_all is not None and existing_all == all_exports:
        return

    content = (
        f'# pylint: disable=too-many-lines\n"""Auto-generated {module_name} module exports."""\n\n'
    )
    content += "\n".join(import_lines) + "\n\n"
    content += "__all__ = [\n"
    for e in all_exports:
        content += f'    "{e}",\n'
    content += "]\n"

    with open(filepath, "w") as f:
        f.write(content)


if __name__ == "__main__":
    lax_subs = ["array", "control_flow", "linalg", "math", "neural_network", "parallel"]
    generate_init(
        "src/ml_switcheroo_compiler/lax/__init__.py", "ml_switcheroo_compiler.lax", lax_subs
    )

    vision_subs = [
        "affine",
        "bbox",
        "color",
        "filtering",
        "interpolation",
        "mixing",
        "transforms",
    ]
    generate_init(
        "src/ml_switcheroo_compiler/ops/vision/__init__.py",
        "ml_switcheroo_compiler.ops.vision",
        vision_subs,
    )

    print("Successfully generated explicit __all__ exports.")
