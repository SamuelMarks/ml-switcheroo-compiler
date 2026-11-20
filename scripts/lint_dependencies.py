"""Linting script to enforce strict third-party dependency and dynamic-import boundaries.

Allowed global third-party dependencies outside tests:
- ml-switcheroo-ir (ml_switcheroo_ir) [Tier 1 Unified IR]
- pyyaml (yaml)
- pillow (PIL)
- pydantic
- cdd / cdd-python
- libcst
- h5py

Allowed backend-specific single external dependencies:
- backends/pytorch: torch
- backends/jax: jax
- backends/mlx: mlx
- backends/cupy: cupy
- backends/dask: dask
- backends/tensorflow: tensorflow
- backends/keras: keras
- backends/numba: numba
- backends/sparse: sparse
- backends/edge: onnx
- backends/metal: Metal
- backends/dpnp: dpnp
- backends/awkward: awkward
- backends/pyarrow_compute: pyarrow
- backends/numpy: numpy, scipy (sole multi-root exception)
"""

from __future__ import annotations

import ast
import os
import sys

STDLIB_MODULES: set[str] = set(sys.builtin_module_names) | {
    "posix",
    "nt",
    "os",
    "sys",
    "time",
    "math",
    "cmath",
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
    "argparse",
    "keyword",
    "pprint",
}

GLOBAL_ALLOWED_THIRD_PARTY: set[str] = {
    "ml_switcheroo_ir",
    "yaml",
    "PIL",
    "pillow",
    "pydantic",
    "cdd",
    "libcst",
    "h5py",
}

ALLOWED_GLOBAL_EXTERNAL: set[str] = GLOBAL_ALLOWED_THIRD_PARTY | {"ml_switcheroo_compiler"}

BACKEND_ALLOWED: dict[str, set[str]] = {
    "pytorch": {"torch"},
    "jax": {"jax"},
    "mlx": {"mlx"},
    "cupy": {"cupy"},
    "cuda": set(),
    "rocm": set(),
    "llvm_cpp": set(),
    "pure_python": set(),
    "common": set(),
    "eager": set(),
    "dask": {"dask"},
    "tensorflow": {"tensorflow"},
    "keras": {"keras"},
    "numpy": {"numpy", "scipy"},
    "numba": {"numba"},
    "sparse": {"sparse"},
    "edge": {"onnx"},
    "metal": {"Metal"},
    "dpnp": {"dpnp"},
    "awkward": {"awkward"},
    "pyarrow_compute": {"pyarrow"},
}


def _is_dynamic_import_call(node: ast.Call) -> bool:
    """Return True if ``node`` is an ``importlib.import_module`` or ``__import__`` call.

    Args:
        node (ast.Call): AST Call node to inspect.

    Returns:
        bool: True when ``node`` invokes ``import_module`` or ``__import__``.
    """
    if isinstance(node.func, ast.Name) and node.func.id in ("__import__", "import_module"):
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    )


def _is_registry_loader_file(rel_path: str) -> bool:
    """Return True when ``rel_path`` is the centralized backend loader ``backends/registry.py``.

    Args:
        rel_path (str): Relative file path inside the scanned root directory.

    Returns:
        bool: True if ``rel_path`` points to ``backends/registry.py``.
    """
    from pathlib import PureWindowsPath

    norm = PureWindowsPath(rel_path).as_posix()
    return norm == "backends/registry.py" or norm.endswith("/backends/registry.py")


def _extract_literal_dynamic_import(
    node: ast.Call,
    file_path: str,
    rel_path: str,
    violations: list[str],
) -> str | None:
    """Validate dynamic import arguments and extract a literal root module name when present.

    Rejects non-literal dynamic ``import_module(var)`` / ``__import__(var)`` calls outside
    ``backends/registry.py``.

    Args:
        node (ast.Call): Dynamic import AST Call node.
        file_path (str): File path for diagnostic reporting.
        rel_path (str): Relative path used to check ``backends/registry.py`` exemption.
        violations (list[str]): Mutable violation list.

    Returns:
        str | None: Top-level module name for literal imports, else ``None``.
    """
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        root_mod = node.args[0].value.lstrip(".").split(".")[0]
        return root_mod if root_mod else None

    if not _is_registry_loader_file(rel_path):
        violations.append(
            f"{file_path}:{node.lineno}: Non-literal dynamic import is forbidden outside 'backends/registry.py'."
        )
    return None


def _record_imported_root(
    mod_name: str,
    file_path: str,
    lineno: int,
    backend_name: str | None,
    backend_allowed: set[str],
    current_root: str,
    backend_observed: dict[str, set[str]],
    violations: list[str],
) -> None:
    """Validate ``mod_name`` and record backend-specific third-party roots.

    Args:
        mod_name (str): Top-level imported module name.
        file_path (str): File path containing the import.
        lineno (int): Source line number.
        backend_name (str | None): Current ``backends/<target>`` subdirectory if any.
        backend_allowed (set[str]): Allowed third-party packages for ``backend_name``.
        current_root (str): Directory containing ``file_path``.
        backend_observed (dict[str, set[str]]): Observed backend-specific third-party roots per backend.
        violations (list[str]): Mutable violation list.
    """
    _check_module_name(mod_name, file_path, lineno, backend_allowed, current_root, violations)
    if (
        backend_name is not None
        and mod_name not in STDLIB_MODULES
        and mod_name not in ALLOWED_GLOBAL_EXTERNAL
        and mod_name in backend_allowed
    ):
        backend_observed.setdefault(backend_name, set()).add(mod_name)


def _scan_ast_imports(
    tree: ast.AST,
    file_path: str,
    rel_path: str,
    backend_name: str | None,
    backend_allowed: set[str],
    root: str,
    backend_observed: dict[str, set[str]],
    violations: list[str],
) -> None:
    """Walk an AST and validate all static imports, dynamic imports, and ``eval`` calls.

    Args:
        tree (ast.AST): Parsed Python module AST.
        file_path (str): Path to the file being scanned.
        rel_path (str): Path relative to the scanned root directory.
        backend_name (str | None): Current ``backends/<target>`` directory name if inside ``backends/``.
        backend_allowed (set[str]): Whitelisted backend-specific top-level module names.
        root (str): Directory containing ``file_path``.
        backend_observed (dict[str, set[str]]): Accumulator of observed backend third-party roots.
        violations (list[str]): Mutable list accumulating violation descriptions.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _record_imported_root(
                    alias.name.split(".")[0],
                    file_path,
                    node.lineno,
                    backend_name,
                    backend_allowed,
                    root,
                    backend_observed,
                    violations,
                )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            _record_imported_root(
                node.module.split(".")[0],
                file_path,
                node.lineno,
                backend_name,
                backend_allowed,
                root,
                backend_observed,
                violations,
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                violations.append(
                    f"{file_path}:{node.lineno}: Prohibited builtin '{node.func.id}()' call; use AST validation "
                    "or file-backed module loading instead."
                )
            if _is_dynamic_import_call(node):
                dyn_mod = _extract_literal_dynamic_import(node, file_path, rel_path, violations)
                if dyn_mod is not None:
                    _record_imported_root(
                        dyn_mod,
                        file_path,
                        node.lineno,
                        backend_name,
                        backend_allowed,
                        root,
                        backend_observed,
                        violations,
                    )


def _validate_single_backend_dependency(
    directory: str,
    backend_observed: dict[str, set[str]],
    violations: list[str],
) -> None:
    """Verify every ``backends/<target>/`` directory imports at most one backend third-party root.

    Permits ``scipy`` alongside ``numpy`` solely inside ``backends/numpy/``.

    Args:
        directory (str): Root directory scanned.
        backend_observed (dict[str, set[str]]): Observed backend third-party roots per backend directory.
        violations (list[str]): Mutable violation list.
    """
    for backend_name, roots in sorted(backend_observed.items()):
        effective_roots = roots - {"scipy"} if backend_name == "numpy" else set(roots)
        if len(effective_roots) > 1:
            backend_dir = os.path.join(directory, "backends", backend_name)
            violations.append(
                f"{backend_dir}: Backend directory '{backend_name}' imports multiple third-party roots: {sorted(roots)}"
            )


def check_dependencies(directory: str) -> list[str]:
    """Scan Python files in ``directory`` and detect dependency or dynamic-import violations.

    Args:
        directory (str): Root source directory to scan.

    Returns:
        list[str]: List of violation messages with file path and offending import details.
    """
    violations: list[str] = []
    backend_observed: dict[str, set[str]] = {}

    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if not file.endswith(".py"):
                continue

            file_path: str = os.path.join(root, file)
            rel_path: str = os.path.relpath(file_path, directory)
            parts: list[str] = rel_path.split(os.sep)

            backend_name: str | None = None
            backend_allowed: set[str] = set()
            if parts[0] == "backends" and len(parts) > 2:
                backend_name = parts[1]
                backend_allowed = BACKEND_ALLOWED.get(backend_name, set())

            try:
                with open(file_path, encoding="utf-8") as f:
                    tree: ast.AST = ast.parse(f.read(), filename=file_path)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue

            _scan_ast_imports(
                tree,
                file_path,
                rel_path,
                backend_name,
                backend_allowed,
                root,
                backend_observed,
                violations,
            )

    _validate_single_backend_dependency(directory, backend_observed, violations)
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
    if any(
        mod_name == prefix or mod_name.startswith(f"{prefix}_") or mod_name.startswith(f"{prefix}-")
        for prefix in ("zero", "zero_zoo")
    ):
        violations.append(
            f"{file_path}:{lineno}: Upward import from '{mod_name}' is strictly forbidden by Tier 2 DAG constraints."
        )
        return
    if mod_name in STDLIB_MODULES or mod_name in ALLOWED_GLOBAL_EXTERNAL or mod_name in backend_allowed:
        return

    top_dir: str = os.path.join("src", "ml_switcheroo_compiler", mod_name)
    top_file: str = os.path.join("src", "ml_switcheroo_compiler", f"{mod_name}.py")
    curr_dir: str = os.path.join(current_root, mod_name)
    curr_file: str = os.path.join(current_root, f"{mod_name}.py")

    if (
        mod_name == "scripts"
        or os.path.exists(top_dir)
        or os.path.exists(top_file)
        or os.path.exists(curr_dir)
        or os.path.exists(curr_file)
    ):
        return

    violations.append(f"{file_path}:{lineno}: Disallowed 3rd-party import '{mod_name}'")


def main() -> int:
    """Execute the dependency isolation linter across source and script files.

    Returns:
        int: Status code 0 on success, 1 if violations are present.
    """
    violations: list[str] = check_dependencies("src/ml_switcheroo_compiler") + check_dependencies("scripts")

    if violations:
        print("Dependency isolation violations detected:")
        for v in violations:
            print(f"  {v}")
        return 1

    print("Dependency isolation check passed: All imports adhere to ecosystem constraints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
