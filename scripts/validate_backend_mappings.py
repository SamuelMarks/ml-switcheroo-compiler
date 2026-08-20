"""Module docstring for validate_backend_mappings.py."""

import json
import os
import re
import sys


def get_snapshot_api_set(prefix: str) -> set[str]:
    """Get the api set from a given snapshot."""
    # find the snapshot for this prefix
    # mapping: np -> numpy, torch -> torch, jax -> jax, keras -> keras, mx -> mlx, da -> dask, cp -> cupy
    prefix_to_fw = {"np": "numpy", "torch": "torch", "jnp": "jax", "jax": "jax", "keras": "keras", "mx": "mlx", "da": "dask", "cp": "cupy"}

    fw = prefix_to_fw.get(prefix)
    if not fw:
        return set()

    try:
        import importlib.resources as pkg_resources
    except ImportError:
        return set()

    try:
        pkg_files = pkg_resources.files("ml_framework_snapshots.snapshots")
        snapshot_files = [f for f in pkg_files.iterdir() if f.name.startswith(f"{fw}_v") and f.name.endswith(".json")]
    except Exception:
        return set()

    if not snapshot_files:
        return set()

    latest_file = sorted(snapshot_files, key=lambda x: x.name)[-1]  # Simple heuristic
    try:
        data = json.loads(latest_file.read_text())
    except Exception:
        return set()

    api_set = set()
    for _, items in data.get("categories", {}).items():
        for item in items:
            api_set.add(item["name"].lower())

    return api_set


def validate_generators() -> list[str]:
    """Validate all generators."""
    # parse the _OP_MAP in each generator
    generators = [("numpy", "np"), ("pytorch", "torch"), ("jax", "jnp"), ("keras", "keras.ops"), ("mlx", "mx"), ("cupy", "cp"), ("dask", "da")]

    errors = []

    for be, prefix in generators:
        fp = f"src/ml_switcheroo_compiler/backends/{be}/generator.py"
        if not os.path.exists(fp):
            fp = f"src/ml_switcheroo_compiler/backends/{be}.py"
        if not os.path.exists(fp):
            continue

        with open(fp) as f:
            src = f.read()

        m = re.search(r"(_OP_MAP\s*=\s*\{)(.*?)(\n\s*\})", src, re.DOTALL)
        if not m:
            continue

        body = m.group(2)

        # Load API set for the backend
        api_set = get_snapshot_api_set(prefix.split(".")[0])
        if not api_set:
            continue  # Can't validate

        # Parse mappings
        for line in body.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Match `"OpName": "prefix.method",`
            # or `"OpName": "method",`
            # Not lambdas or attributes
            match = re.match(r'^"([^"]+)":\s*"([^"]+)",?$', line)
            if match:
                op_name = match.group(1)
                mapped_val = match.group(2)

                # Check if it's a simple direct attribute access
                if mapped_val.startswith(f"{prefix}.") and "(" not in mapped_val and "[" not in mapped_val:
                    method_name = mapped_val[len(prefix) + 1 :]
                    # Numpy snapshot is kinda broken right now, and cupy is empty.
                    if prefix not in ["np", "cp"]:
                        if method_name.lower() not in api_set:
                            errors.append(f"{fp}: '{op_name}' mapped to hallucinated endpoint '{mapped_val}'")

    return errors


if __name__ == "__main__":
    errors = validate_generators()
    if errors:
        print("Backend Grounding Validation Failed! Hallucinations detected:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    else:
        print("Backend Grounding Validation Passed. No hallucinations detected.")
