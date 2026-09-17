"""Linting script to enforce strict third-party dependency boundaries.

Allowed global third-party dependencies outside tests:
- pyyaml (yaml)
- pillow (PIL)
- pydantic
- cdd / cdd-python
- libcst
- h5py
- numpy (standard foundation)

Allowed backend-specific single external dependencies:
- backends/pytorch: torch
- backends/jax: jax
- backends/mlx: mlx
- backends/cupy: cupy
- backends/dask: dask
- backends/tensorflow: tensorflow
- backends/keras: keras
- backends/numpy (and backends/eager): scipy
"""

import ast
import os
import sys

STDLIB_MODULES: set[str] = (
    set(sys.builtin_module_names)
    | set(getattr(sys, "stdlib_module_names", []))
    | {
        "posix",
        "nt",
        "os",
        "sys",
        "time",
        "math",
        "re",
        "json",
        "typing",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "dataclasses",
        "enum",
        "uuid",
        "abc",
        "contextlib",
        "copy",
        "io",
        "inspect",
        "shutil",
        "tempfile",
        "ctypes",
        "struct",
        "zlib",
        "hashlib",
        "resource",
        "threading",
        "asyncio",
        "socket",
        "pickle",
        "zipfile",
        "tarfile",
        "csv",
        "multiprocessing",
        "importlib",
        "builtins",
        "__future__",
        "glob",
        "urllib",
        "gc",
        "unittest",
        "logging",
        "warnings",
        "base64",
        "ast",
        "subprocess",
        "gzip",
        "contextvars",
        "queue",
        "select",
        "http",
        "signal",
        "numbers",
        "operator",
        "decimal",
        "fractions",
        "weakref",
        "types",
        "traceback",
        "linecache",
        "tokenize",
        "token",
        "dis",
        "platform",
        "site",
        "heapq",
        "bisect",
        "array",
        "random",
        "string",
    }
)

ALLOWED_GLOBAL_EXTERNAL: set[str] = {
    "yaml",
    "PIL",
    "pydantic",
    "cdd",
    "libcst",
    "h5py",
    "numpy",
    "ml_switcheroo_compiler",
    "ml_switcheroo_ir",
}

BACKEND_ALLOWED: dict[str, set[str]] = {
    "pytorch": {"torch"},
    "torch": {"torch"},
    "jax": {"jax"},
    "mlx": {"mlx"},
    "cupy": {"cupy"},
    "dask": {"dask"},
    "tensorflow": {"tensorflow", "tf"},
    "keras": {"keras", "keras_core"},
    "numpy": {"scipy"},
    "eager": {"scipy"},
    "common": {"scipy"},
    "numba": {"numba"},
    "sparse": {"sparse", "scipy"},
}

FORBIDDEN_UPWARD_PREFIXES: tuple[str, ...] = ("zero_", "zero_zoo", "zero")


def check_dependencies(directory: str) -> list[str]:
    """Scan Python files and detect any non-whitelisted third-party imports.

    Args:
        directory (str): Root source directory to scan.

    Returns:
        list[str]: List of violation messages with file path and imported module name.
    """
    violations: list[str] = []

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path: str = os.path.join(root, file)
            rel_path: str = os.path.relpath(file_path, directory)
            parts: list[str] = rel_path.split(os.sep)

            backend_allowed: set[str] = set()
            if parts[0] == "backends" and len(parts) > 1:
                bname: str = parts[1]
                backend_allowed = BACKEND_ALLOWED.get(bname, set())

            try:
                with open(file_path, encoding="utf-8") as f:
                    tree: ast.AST = ast.parse(f.read(), filename=file_path)
            except Exception:
                continue

            for node in ast.walk(tree):
                mod_name: str = ""
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod_name = alias.name.split(".")[0]
                        _check_module_name(mod_name, file_path, node.lineno, backend_allowed, root, violations)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mod_name = node.module.split(".")[0]
                    _check_module_name(mod_name, file_path, node.lineno, backend_allowed, root, violations)

    return violations


def _check_module_name(
    mod_name: str,
    file_path: str,
    lineno: int,
    backend_allowed: set[str],
    current_root: str,
    violations: list[str],
) -> None:
    """Validate an imported top-level module name against whitelist schemas.

    Args:
        mod_name (str): Top-level module name.
        file_path (str): Path to file containing import.
        lineno (int): Line number of import statement.
        backend_allowed (set[str]): Framework imports whitelisted for current backend directory.
        current_root (str): Current directory for checking local module existence.
        violations (list[str]): Output list to record violations.
    """
    if not mod_name:
        return
    if any(mod_name == prefix or mod_name.startswith(f"{prefix}_") or mod_name.startswith(f"{prefix}-") for prefix in ("zero", "zero_zoo")):
        violations.append(f"{file_path}:{lineno}: Upward import from '{mod_name}' is strictly forbidden by Tier 2 DAG constraints.")
        return
    if mod_name in STDLIB_MODULES or mod_name in ALLOWED_GLOBAL_EXTERNAL or mod_name in backend_allowed:
        return

    # Check if module exists locally in package
    top_dir: str = os.path.join("src", "ml_switcheroo_compiler", mod_name)
    top_file: str = os.path.join("src", "ml_switcheroo_compiler", f"{mod_name}.py")
    curr_dir: str = os.path.join(current_root, mod_name)
    curr_file: str = os.path.join(current_root, f"{mod_name}.py")

    if os.path.exists(top_dir) or os.path.exists(top_file) or os.path.exists(curr_dir) or os.path.exists(curr_file):
        return

    violations.append(f"{file_path}:{lineno}: Disallowed 3rd-party import '{mod_name}'")


def main() -> int:
    """Execute the dependency isolation linter across source files.

    Returns:
        int: Status code 0 on success, 1 if violations are present.
    """
    directory_to_scan: str = "src/ml_switcheroo_compiler"
    violations: list[str] = check_dependencies(directory_to_scan)

    if violations:
        print("Dependency isolation violations detected:")
        for v in violations:
            print(f"  {v}")
        return 1

    print("Dependency isolation check passed: All imports adhere to ecosystem constraints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
