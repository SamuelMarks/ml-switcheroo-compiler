"""Enforce backend isolation rules."""

import ast
import os
import sys

FORBIDDEN_IMPORTS = {
    "src/ml_switcheroo_compiler/backends/cupy.py": {
        "numpy",
        "torch",
        "tensorflow",
        "keras",
        "dask",
        "jax",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/numpy.py": {
        "cupy",
        "torch",
        "tensorflow",
        "keras",
        "dask",
        "jax",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/dask.py": {
        "cupy",
        "torch",
        "tensorflow",
        "keras",
        "numpy",
        "jax",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/pytorch.py": {
        "numpy",
        "cupy",
        "tensorflow",
        "keras",
        "dask",
        "jax",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/keras.py": {
        "numpy",
        "cupy",
        "torch",
        "tensorflow",
        "dask",
        "jax",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/tensorflow.py": {
        "numpy",
        "cupy",
        "torch",
        "keras",
        "dask",
        "jax",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/jax.py": {
        "numpy",
        "cupy",
        "torch",
        "tensorflow",
        "keras",
        "dask",
        "mlx",
    },
    "src/ml_switcheroo_compiler/backends/mlx.py": {
        "numpy",
        "cupy",
        "torch",
        "tensorflow",
        "keras",
        "dask",
        "jax",
    },
}


def check_file(file_path: str) -> bool:
    """Check a file for forbidden imports.

    Args:
        file_path (str): The file path to check.

    Returns:
        bool: True if file is valid, False otherwise.
    """
    if not os.path.exists(file_path):
        return True

    # Convert to standard format
    file_path = os.path.normpath(file_path)

    forbidden = FORBIDDEN_IMPORTS.get(file_path, set())
    if (
        "src/ml_switcheroo_compiler/core/" in file_path
        or "src/ml_switcheroo_compiler/ops/" in file_path
    ):
        forbidden = {"numpy", "cupy", "torch", "tensorflow", "keras", "dask", "jax", "mlx"}

    if not forbidden:
        return True

    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False

    has_error = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in forbidden:
                    print(f"[{file_path}:{node.lineno}] Forbidden import detected: {alias.name}")
                    has_error = True
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module in forbidden:
                    msg = f"[{file_path}:{node.lineno}] Forbidden import detected: "
                    msg += f"from {node.module} import ..."
                    print(msg)
                    has_error = True

    return not has_error


if __name__ == "__main__":
    files_to_check = sys.argv[1:]
    success = True
    for file_path in files_to_check:
        if not check_file(file_path):
            success = False
    if not success:
        sys.exit(1)
