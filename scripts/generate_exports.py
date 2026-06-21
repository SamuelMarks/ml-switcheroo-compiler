"""Build script to dynamically regenerate __all__ literal string lists."""

import ast
import importlib
from typing import Optional


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
    all_exports = []
    import_lines = []
    imported_symbols = set()

    for submod in submodules:
        modname = f"{module_name}.{submod}"
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            print(f"Failed to import {modname}: {e}")
            continue

        if hasattr(mod, "__all__"):
            exports = sorted(list(set(mod.__all__)))
        else:
            exports = sorted(list(set([n for n in dir(mod) if not n.startswith("_")])))

        unique_exports = [e for e in exports if e not in imported_symbols]
        if unique_exports:
            import_lines.append(f"from {modname} import (")
            for e in unique_exports:
                import_lines.append(f"    {e},")
            import_lines.append(")")
            imported_symbols.update(unique_exports)
            all_exports.extend(unique_exports)

    if extra_imports:
        for modname, exports in extra_imports:
            unique_exports = [e for e in sorted(set(exports)) if e not in imported_symbols]
            if unique_exports:
                import_lines.append(f"from {modname} import (")
                for e in unique_exports:
                    import_lines.append(f"    {e},")
                import_lines.append(")")
                imported_symbols.update(unique_exports)
                all_exports.extend(unique_exports)

    all_exports = sorted(list(set(all_exports)))

    existing_all = None
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            existing_all = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
    except Exception:
        pass

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
    ops_subs = [
        "aliases",
        "audio",
        "base",
        "binary",
        "creation",
        "distributed",
        "linalg",
        "nn",
        "normalization",
        "reductions",
        "shape",
        "text",
        "unary",
        "vision",
        "state",
    ]
    ops_extra = [
        (
            "ml_switcheroo_compiler.ops.creation.frontend",
            [
                "array",
                "asarray",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
                "full",
                "full_like",
                "eye",
                "diag",
                "empty",
                "empty_like",
                "linspace",
                "arange",
            ],
        ),
        (
            "ml_switcheroo_compiler.ops.vision.ops",
            [
                "ResizeBicubic",
                "ResizeBilinear",
                "ResizeNearest",
                "ResizeLanczos3",
                "IoU",
                "NonMaxSuppression",
                "ExtractBoundingBoxes",
                "PerspectiveTransform",
                "ElasticTransform",
                "GaussianBlur",
                "MedianFilter",
            ],
        ),
    ]
    generate_init(
        "src/ml_switcheroo_compiler/ops/__init__.py",
        "ml_switcheroo_compiler.ops",
        ops_subs,
        ops_extra,
    )

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
