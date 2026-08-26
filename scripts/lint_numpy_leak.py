"""Linting script to enforce the No NumPy Fallback rule and Architectural Boundaries."""

import glob
import re
import sys


def check_for_numpy_leaks(directory: str) -> list[str]:
    """Check for numpy leaks in backends (excluding numpy backend and eager execution).

    Args:
        directory (str): The root directory to scan.

    Returns:
        list[str]: A list of violation messages.
    """
    violations: list[str] = []

    files_to_check: list[str] = glob.glob(directory + "/backends/**/*.py", recursive=True)

    # Explicitly check root level core tracing files
    files_to_check.extend(glob.glob(f"{directory}/grad/**/*.py", recursive=True))

    for filepath in set(files_to_check):
        # Whitelisted directories/files
        if "backends/numpy" in filepath or "generator_mixins.py" in filepath:
            continue
        if "backends/eager" in filepath:
            # Eager evaluators are allowed to use numpy for unbacked/fallback math
            continue

        with open(filepath) as f:
            for i, line in enumerate(f):
                # Ignore comments
                if line.strip().startswith("#"):
                    continue

                # Strip inline comments to prevent false positives
                code_line: str = line.split("#")[0].strip()

                has_import: bool = "import numpy" in code_line or "from numpy" in code_line
                has_np: bool = bool(re.search(r"\bnp\.", code_line))

                if has_import or has_np:
                    # Ignore .numpy() attribute calls, common in PyTorch/TF tests
                    if re.search(r"\w+\.numpy\(\)", code_line):
                        continue

                    # Ignore specific documented exceptions
                    if "np.gradient" in code_line and "torch.gradient" in line:
                        continue

                    violations.append(f"{filepath}:{i + 1}: {code_line}")

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

    for filepath in set(files_to_check):
        with open(filepath) as f:
            for i, line in enumerate(f):
                if line.strip().startswith("#"):
                    continue
                code_line: str = line.split("#")[0].strip()

                # Check if it imports from ml_switcheroo_compiler.backends
                if "ml_switcheroo_compiler.backends" in code_line:
                    # Allow registry
                    if "ml_switcheroo_compiler.backends.registry" in code_line:
                        continue
                    # Allow linker source_ast_ref?
                    if "ml_switcheroo_compiler.backends.linker" in code_line:
                        continue

                    violations.append(f"{filepath}:{i + 1}: {code_line}")

    return violations


def main() -> int:
    """Run linting.

    Returns:
        int: 0 if success, 1 if failed.
    """
    directory_to_check: str = "src/ml_switcheroo_compiler"
    numpy_violations: list[str] = check_for_numpy_leaks(directory_to_check)
    arch_violations: list[str] = check_for_architectural_imports(directory_to_check)

    if numpy_violations or arch_violations:
        if numpy_violations:
            print("NumPy Leak Linting failed. Found restricted numpy references in backends:")
            for v in numpy_violations:
                print(v)
        if arch_violations:
            print("Architectural Boundaries failed. Found restricted imports from backends:")
            for v_arch in arch_violations:
                print(v_arch)
        return 1

    print("NumPy Leak and Architectural Boundaries Linting passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
