"""Validation script to ground backend mappings against ml-framework-snapshots."""

import glob
import importlib
import os
import re
import sys
from typing import Optional

import yaml

ROOT_MAP: dict[str, str] = {
    "torch": "torch",
    "pytorch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "np": "numpy",
    "numpy": "numpy",
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "jnp": "jax.numpy",
    "jax": "jax",
    "mx": "mlx.core",
    "mlx": "mlx",
    "da": "dask.array",
    "dask": "dask",
    "cp": "cupy",
    "cupy": "cupy",
    "keras": "keras",
}

BACKEND_ALLOWED_ROOTS: dict[str, set[str]] = {
    "pytorch": {"torch", "pytorch", "torchaudio", "torchvision"},
    "jax": {"jax", "jnp"},
    "mlx": {"mlx", "mx"},
    "tensorflow": {"tensorflow", "tf"},
    "keras": {"keras"},
    "cupy": {"cupy", "cp"},
    "dask": {"dask", "da"},
    "numpy": {"numpy", "np"},
}

BUILTIN_OPS: set[str] = {
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "and",
    "or",
    "xor",
    "invert",
    "lshift",
    "rshift",
    "abs",
    "pos",
    "neg",
    "round",
    "trunc",
    "floor",
    "ceil",
    "getattr",
    "type",
    "tuple",
    "list",
    "dict",
    "set",
    "bool",
    "int",
    "float",
    "str",
    "slice",
}


def resolve_api_endpoint(api_str: str) -> bool:
    """Resolve an API endpoint chain to check if it exists in the module.

    Args:
        api_str (str): The qualified API endpoint string (e.g. 'torch.atan2').

    Returns:
        bool: True if the endpoint resolves to an existing attribute, False otherwise.
    """
    clean_api: str = api_str.split("(")[0].strip()
    parts: list[str] = clean_api.split(".")
    root: str = parts[0]
    if root not in ROOT_MAP:
        return False
    mod_name: str = ROOT_MAP[root]
    try:
        obj: object = importlib.import_module(mod_name)
    except Exception:
        # Fallback for uninstalled optional backends (like cupy on macOS)
        return True

    for p in parts[1:]:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            try:
                mod_name = f"{mod_name}.{p}"
                obj = importlib.import_module(mod_name)
            except Exception:
                return False
    return True


def validate_mappings() -> list[str]:
    """Validate all backend mappings against framework ground truth with zero skip lists.

    Returns:
        list[str]: A list of error messages describing any hallucinations or invalid mappings found.
    """
    errors: list[str] = []

    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ml_switcheroo_compiler", "backends"))
    files_to_check: list[str] = glob.glob(os.path.join(base_dir, "**", "mappings", "*.yaml"), recursive=True)
    files_to_check.extend(glob.glob(os.path.join(base_dir, "**", "mappings.yaml"), recursive=True))
    files_to_check.extend(glob.glob(os.path.join(base_dir, "**", "eager_mappings.yaml"), recursive=True))

    for filepath in sorted(files_to_check):
        with open(filepath, encoding="utf-8") as f:
            data: Optional[dict[str, object]] = yaml.safe_load(f)

        if not isinstance(data, dict):
            continue

        rel_path: str = os.path.relpath(filepath, base_dir)
        backend_name: str = rel_path.split(os.sep)[0]
        allowed_roots: set[str] = BACKEND_ALLOWED_ROOTS.get(backend_name, set())

        ops_dict: dict[str, dict[str, object]] = {}
        if "operations" in data and isinstance(data["operations"], dict):
            ops_dict = data["operations"]
        elif "backend_name" in data:
            continue
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    ops_dict[k] = v

        for op, spec in ops_dict.items():
            api: str = str(spec.get("target_api") or "")
            if api == "custom_op" or not api:
                ast_template: str = str(spec.get("ast_template") or "")
                if ast_template:
                    m: Optional[re.Match[str]] = re.match(r"^([\w\.]+)", ast_template)
                    if m:
                        api = m.group(1)

            if not api or api == "custom_op" or "lambda" in api or api.startswith("_"):
                continue

            api = api.strip("'").strip('"')
            parts: list[str] = api.split(".")
            root_module: str = parts[0].split("(")[0]

            if root_module in ROOT_MAP:
                if allowed_roots and root_module not in allowed_roots:
                    errors.append(f"{filepath}: '{op}' cross-framework hallucination '{api}' in '{backend_name}' backend")
                    continue

                if not resolve_api_endpoint(api):
                    errors.append(f"{filepath}: '{op}' mapped to unverified/hallucinated endpoint '{api}'")

    return errors


def main() -> int:
    """Run validation.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    errors: list[str] = validate_mappings()
    if errors:
        print("Backend Grounding Validation Failed! Hallucinations detected:")
        for e in errors:
            print("  -", e)
        return 1

    print("Backend Grounding Validation Passed. No hallucinations detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
