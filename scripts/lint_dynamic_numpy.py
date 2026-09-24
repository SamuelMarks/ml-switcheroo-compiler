"""Pre-commit hook guarding against dynamic or indirect imports of numpy outside backends/numpy."""

import ast
import os
import sys

ALLOWED_MODULE_PATTERNS: tuple[str, ...] = (
    os.path.join("backends", "numpy"),
    os.path.join("backends", "eager"),
    os.path.join("backends", "metal"),
    os.path.join("backends", "awkward"),
    os.path.join("backends", "pyarrow_compute"),
    os.path.join("backends", "dpnp"),
)


def check_for_prohibited_numpy_imports(directory: str) -> list[str]:
    """Scan Python files for dynamic numpy import calls like importlib.import_module('numpy').

    Args:
        directory (str): The root source directory to scan.

    Returns:
        list[str]: Violations list formatted as file:line.
    """
    violations: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path: str = os.path.join(root, file)
            if any(pattern in file_path for pattern in ALLOWED_MODULE_PATTERNS):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content: str = f.read()
                    tree: ast.AST = ast.parse(content, filename=file_path)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    is_dynamic_import: bool = False
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                        is_dynamic_import = True
                    elif isinstance(node.func, ast.Name) and node.func.id in ("import_module", "__import__"):
                        is_dynamic_import = True

                    if is_dynamic_import and node.args:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                            if first_arg.value == "numpy" or first_arg.value.startswith("numpy."):
                                violations.append(f"{file_path}:{node.lineno}: Prohibited dynamic numpy import '{first_arg.value}' outside ml-switcheroo-compiler numpy/scipy module.")
    return violations


def main() -> int:
    """Run dynamic numpy import guard.

    Returns:
        int: 0 if passed, 1 if violations found.
    """
    directory: str = "src/ml_switcheroo_compiler"
    violations: list[str] = check_for_prohibited_numpy_imports(directory)
    if violations:
        print("Prohibited dynamic numpy imports detected:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("Dynamic numpy import guard passed: No unauthorized dynamic numpy imports found.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
