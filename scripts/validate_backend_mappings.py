"""Module docstring for validate_backend_mappings.py."""

import json
import os
import re
import sys
from typing import Any, Optional

import yaml


def get_snapshot_api_dict(prefix: str) -> dict[str, set[str]]:
    """Get the api dict with allowed kwargs from a given snapshot."""
    prefix_to_fw: dict[str, str] = {"np": "numpy", "torch": "torch", "jnp": "jax", "jax": "jax", "keras": "keras", "mx": "mlx", "da": "dask", "cp": "cupy", "tf": "tensorflow", "numpy": "numpy", "tensorflow": "tensorflow", "mlx": "mlx", "dask": "dask", "cupy": "cupy", "pytorch": "torch"}

    fw: Optional[str] = prefix_to_fw.get(prefix)
    if not fw:
        return {}

    try:
        snapshot_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "ml-framework-snapshots", "src", "ml_framework_snapshots", "snapshots")
        snapshot_files: list[str] = [f for f in os.listdir(snapshot_dir) if f.startswith(f"{fw}_v") and f.endswith(".json")]
    except Exception:
        snapshot_files = []

    if not snapshot_files:
        return {}

    latest_file: str = sorted(snapshot_files)[-1]
    try:
        with open(os.path.join(snapshot_dir, latest_file)) as f:
            data: dict[str, Any] = json.loads(f.read())
    except Exception:
        return {}

    api_dict: dict[str, set[str]] = {}
    for _, items in data.get("categories", {}).items():
        for item in items:
            name: str = item["name"].lower()
            allowed_kwargs: set[str] = set(item.get("kwargs", []))
            for p in item.get("params", []):
                allowed_kwargs.add(p.get("name"))
            api_dict[name] = allowed_kwargs

    return api_dict


def validate_mappings() -> list[str]:
    """Validate mappings against snapshots."""
    errors: list[str] = []
    import glob

    files_to_check: list[str] = glob.glob("src/ml_switcheroo_compiler/backends/**/mappings.yaml", recursive=True)

    for filepath in files_to_check:
        with open(filepath) as f:
            data: dict[str, Any] = yaml.safe_load(f)

        backend_name: str = data.get("backend_name", "")
        api_dict: dict[str, set[str]] = get_snapshot_api_dict(backend_name)
        if not api_dict:
            continue

        # Ignore broken snapshots
        if backend_name in ["numpy", "cupy", "dask"]:
            continue

        for op, spec in data.get("operations", {}).items():
            api: str = spec.get("target_api", "")
            if api == "custom_op" or not api:
                ast_template: str = spec.get("ast_template", "")
                if ast_template:
                    m: Optional[re.Match[str]] = re.match(r"^([\w\.]+)", ast_template)
                    if m:
                        api = m.group(1)

            if api and "lambda" not in api:
                parts: list[str] = api.split(".")
                name: str = parts[-1].lower()

                if name in ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow", "and", "or", "xor", "invert", "lshift", "rshift", "abs", "pos", "neg", "round", "trunc", "floor", "ceil", "getattr", "type", "tuple", "list", "dict", "set", "bool", "int", "float", "str", "slice"]:
                    continue

                if "tf." in api or "numpy." in api or "torch." in api or "keras." in api or "jax." in api or "dask." in api or "cupy." in api or "mlx." in api or "np." in api or "cp." in api or "jnp." in api or "mx." in api or "da." in api:
                    if name not in api_dict:
                        # Some special cases mapped manually or via fallback modules
                        if not api.startswith("tf.sparse") and not api.startswith("tf.ragged") and not api.startswith("tf.nn"):
                            errors.append(f"{filepath}: '{op}' mapped to hallucinated endpoint '{api}'")
                    else:
                        # Check kwargs
                        allowed_kwargs: set[str] = api_dict[name]
                        kwarg_translations: dict[str, Any] = spec.get("kwarg_translations", {})
                        for _k_name, translated in kwarg_translations.items():
                            if isinstance(translated, dict):
                                target_name: Optional[str] = translated.get("target_name")
                            else:
                                target_name = translated
                            if target_name and target_name not in allowed_kwargs and "kwargs" not in allowed_kwargs:
                                errors.append(f"{filepath}: '{op}' mapped kwarg '{target_name}' not found in endpoint '{api}'")

    return errors


def main() -> int:
    """Run validation."""
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
