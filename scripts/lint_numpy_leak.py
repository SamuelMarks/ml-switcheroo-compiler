"""Linting script to enforce the No NumPy Fallback rule and Architectural Boundaries."""

from __future__ import annotations

import glob
import re
import sys


def _scan_file_for_np_refs(filepath: str) -> list[str]:
    """Scan a single file for prohibited 'np.' references.

    Args:
        filepath (str): File path to inspect.

    Returns:
        list[str]: Detected violation strings.
    """
    hits: list[str] = []
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            continue
        code_line = line.split("#")[0].strip()
        if re.search(r"\bnp\.", code_line):
            if re.search(r"\w+\.numpy\(\)", code_line) or ("np.gradient" in code_line and "torch.gradient" in line):
                continue
            hits.append(f"{filepath}:{i + 1}: {code_line}")
        if re.search(r"\b(_np_mod|_scipy_mod|numpy_mod)\b", code_line):
            hits.append(f"{filepath}:{i + 1}: {code_line}")
    return hits


def check_for_numpy_leaks(directory: str) -> list[str]:
    """Check for numpy leaks across the compiler codebase.

    Args:
        directory (str): The root directory to scan.

    Returns:
        list[str]: A list of violation messages.
    """
    violations: list[str] = []

    all_files: list[str] = glob.glob(f"{directory}/**/*.py", recursive=True)
    import_pattern = re.compile(r"^\s*(import\s+numpy|from\s+numpy)\b")

    for filepath in sorted(set(all_files)):
        if "backends/numpy" in filepath:
            continue
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            code_line: str = line.split("#")[0].strip()
            if import_pattern.search(code_line):
                violations.append(f"{filepath}:{i + 1}: {code_line}")

    exempt_dirs = ("backends/numpy",)
    for filepath in sorted(set(all_files)):
        if not any(exempt in filepath for exempt in exempt_dirs):
            violations.extend(_scan_file_for_np_refs(filepath))

    return violations


def check_for_architectural_imports(directory: str) -> list[str]:
    """Check that core/, ir/, and ops/ never import from backends/ except registry.

    Args:
        directory (str): The root directory to scan.

    Returns:
        list[str]: A list of violation messages.
    """
    violations: list[str] = []

    files_to_check: list[str] = []
    for d in ["core", "ir", "ops", "transforms"]:
        files_to_check.extend(glob.glob(directory + f"/{d}/**/*.py", recursive=True))

    import_stmt_pattern = re.compile(r"\b(?:from|import)\s+ml_switcheroo_compiler\.backends\b")

    for filepath in sorted(set(files_to_check)):
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                continue
            code_line: str = line.split("#")[0].strip()

            # Check if it imports from ml_switcheroo_compiler.backends
            if import_stmt_pattern.search(code_line):
                # Allow registry
                if "ml_switcheroo_compiler.backends.registry" in code_line:
                    continue
                # Allow linker source_ast_ref?
                if "ml_switcheroo_compiler.backends.linker" in code_line:
                    continue

                violations.append(f"{filepath}:{i + 1}: {code_line}")

    return violations


def check_for_third_party_leaks(directory: str) -> list[str]:
    """Check that no unapproved third-party dependencies are imported outside backend directories.

    Args:
        directory (str): The root source directory to scan.

    Returns:
        list[str]: A list of violation messages.
    """
    try:
        from scripts.lint_dependencies import check_dependencies
    except ImportError:
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.lint_dependencies import check_dependencies

    return check_dependencies(directory)


def main() -> int:
    """Run linting.

    Returns:
        int: 0 if success, 1 if failed.
    """
    directory_to_check: str = "src/ml_switcheroo_compiler"
    numpy_violations: list[str] = check_for_numpy_leaks(directory_to_check)
    arch_violations: list[str] = check_for_architectural_imports(directory_to_check)
    third_party_violations: list[str] = check_for_third_party_leaks(directory_to_check)

    if numpy_violations or arch_violations or third_party_violations:
        if numpy_violations:
            print("NumPy Leak Linting failed. Found restricted numpy references in backends:")
            for v in numpy_violations:
                print(v)
        if arch_violations:
            print("Architectural Boundaries failed. Found restricted imports from backends:")
            for v_arch in arch_violations:
                print(v_arch)
        if third_party_violations:
            print("Third-Party Dependency Isolation failed. Found unauthorized external imports:")
            for v_dep in third_party_violations:
                print(v_dep)
        return 1

    print("NumPy Leak, Third-Party Isolation, and Architectural Boundaries Linting passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
