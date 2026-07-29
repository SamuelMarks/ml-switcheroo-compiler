"""Linting script to enforce the No NumPy Fallback rule and Architectural Boundaries."""

import glob
import re
import sys


def check_for_numpy_leaks(directory: str) -> list:
    """Check for numpy leaks in backends (excluding numpy backend and eager execution)."""
    violations = []

    files_to_check = glob.glob(directory + "/backends/**/*.py", recursive=True)

    # Explicitly check root level core tracing files
    files_to_check.append(f"{directory}/grad.py")

    for filepath in set(files_to_check):
        # Whitelisted directories/files
        if "backends/numpy" in filepath:
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
                code_line = line.split("#")[0].strip()

                has_import = "import numpy" in code_line or "from numpy" in code_line
                has_np = re.search(r"\bnp\.", code_line)

                if has_import or has_np:
                    # Ignore .numpy() attribute calls, common in PyTorch/TF tests
                    if re.search(r"\w+\.numpy\(\)", code_line):
                        continue

                    # Ignore specific documented exceptions
                    if "np.gradient" in code_line and "torch.gradient" in line:
                        continue

                    violations.append(f"{filepath}:{i + 1}: {code_line}")

    return violations


def check_for_architectural_imports(directory: str) -> list:
    """Check that core/, ir/, and ops/ never import from backends/ except registry."""
    violations = []

    files_to_check = []
    for d in ["core", "ir", "ops", "transforms"]:
        files_to_check.extend(glob.glob(directory + f"/{d}/**/*.py", recursive=True))

    for filepath in set(files_to_check):
        with open(filepath) as f:
            for i, line in enumerate(f):
                if line.strip().startswith("#"):
                    continue
                code_line = line.split("#")[0].strip()

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


if __name__ == "__main__":
    directory = "src/ml_switcheroo_compiler"
    numpy_violations = check_for_numpy_leaks(directory)
    arch_violations = check_for_architectural_imports(directory)

    if numpy_violations or arch_violations:
        if numpy_violations:
            print("NumPy Leak Linting failed. Found restricted numpy references in backends:")
            for v in numpy_violations:
                print(v)
        if arch_violations:
            print("Architectural Boundaries failed. Found restricted imports from backends:")
            for v in arch_violations:
                print(v)
        sys.exit(1)
    else:
        print("NumPy Leak and Architectural Boundaries Linting passed.")
        sys.exit(0)
