"""Generate .pyi type stubs from ml-framework-snapshots."""

import json
import os


def generate_stubs():
    """Generate stubs."""
    prefix_to_fw = {"numpy": "numpy", "pytorch": "torch", "jax": "jax", "keras": "keras", "mlx": "mlx", "dask": "dask", "cupy": "cupy", "tensorflow": "tensorflow"}

    snapshot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ml-framework-snapshots", "src", "ml_framework_snapshots", "snapshots")
    if not os.path.exists(snapshot_dir):
        print(f"Snapshot directory not found: {snapshot_dir}")
        return

    for be_name, fw in prefix_to_fw.items():
        snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith(f"{fw}_v") and f.endswith(".json")]
        if not snapshot_files:
            continue

        latest_file = sorted(snapshot_files)[-1]
        with open(os.path.join(snapshot_dir, latest_file)) as f:
            data = json.loads(f.read())

        # Collect signatures
        out_lines = ['"""Auto-generated type stubs from snapshot."""', "from typing import Any", ""]

        # very simple flat generation
        # ideally we build a module tree, but for mypy grounding in eagerly dispatched code, a flat set of functions is a good start
        # we will generate this directly into the backend directory
        stub_path = f"src/ml_switcheroo_compiler/backends/{be_name}/stub.pyi"

        # let's just make sure we are not overwriting important stuff
        if not os.path.exists(os.path.dirname(stub_path)):
            continue

        funcs = []
        for _, items in data.get("categories", {}).items():
            for item in items:
                name = item["name"]
                # just generate `def <name>(*args: Any, **kwargs: Any) -> Any: ...`
                if item["kind"] == "function":
                    funcs.append(name)

        # filter invalid names
        valid_funcs = set()
        for f in funcs:
            if f.isidentifier() and not f.startswith("__"):
                valid_funcs.add(f)

        for f in sorted(valid_funcs):
            out_lines.append(f"def {f}(*args: Any, **kwargs: Any) -> Any: ...")

        with open(stub_path, "w") as f:
            f.write("\n".join(out_lines))

        print(f"Generated {stub_path} with {len(valid_funcs)} stubs.")


if __name__ == "__main__":
    generate_stubs()
