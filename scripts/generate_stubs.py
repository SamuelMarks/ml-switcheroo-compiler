"""Generate .pyi type stubs from ml-framework-snapshots."""

import json
import os
import typing


def generate_stubs(snapshot_dir: typing.Optional[str] = None, out_base_dir: str = "src/ml_switcheroo_compiler/backends") -> None:
    """Generate stubs.

    Args:
        snapshot_dir: Optional override for snapshot directory.
        out_base_dir: Optional override for the output base directory.

    Returns:
        None
    """
    prefix_to_fw: dict[str, str] = {"numpy": "numpy", "pytorch": "torch", "jax": "jax", "keras": "keras", "mlx": "mlx", "dask": "dask", "cupy": "cupy", "tensorflow": "tensorflow"}

    if snapshot_dir is None:
        snapshot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ml-framework-snapshots", "src", "ml_framework_snapshots", "snapshots")

    if not os.path.exists(snapshot_dir):
        print(f"Snapshot directory not found: {snapshot_dir}")
        return

    for be_name, fw in prefix_to_fw.items():
        snapshot_files: list[str] = [f for f in os.listdir(snapshot_dir) if f.startswith(f"{fw}_v") and f.endswith(".json")]
        if not snapshot_files:
            continue

        latest_file: str = sorted(snapshot_files)[-1]
        with open(os.path.join(snapshot_dir, latest_file)) as f:
            data = json.loads(f.read())

        # Collect signatures
        out_lines: list[str] = ['"""Auto-generated type stubs from snapshot."""', "class Tensor: ...", ""]

        # very simple flat generation
        # ideally we build a module tree, but for mypy grounding in eagerly dispatched code, a flat set of functions is a good start
        # we will generate this directly into the backend directory
        stub_path: str = os.path.join(out_base_dir, be_name, "stub.pyi")

        # let's just make sure we are not overwriting important stuff
        if not os.path.exists(os.path.dirname(stub_path)):
            continue

        funcs: list[str] = []
        categories = data.get("categories", {})
        if isinstance(categories, dict):
            for _, items in categories.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            name = item.get("name")
                            kind = item.get("kind")
                            # just generate `def <name>(*args: Tensor, **kwargs: Tensor) -> Tensor: ...`
                            if kind == "function" and isinstance(name, str):
                                funcs.append(name)

        # filter invalid names
        valid_funcs: set[str] = set()
        for fn in funcs:
            if fn.isidentifier() and not fn.startswith("__"):
                valid_funcs.add(fn)

        for fn in sorted(valid_funcs):
            out_lines.append(f"def {fn}(*args: Tensor, **kwargs: Tensor) -> Tensor: ...")

        with open(stub_path, "w") as f_out:
            f_out.write("\n".join(out_lines))

        print(f"Generated {stub_path} with {len(valid_funcs)} stubs.")


def main() -> None:
    """Entry point.

    Returns:
        None
    """
    generate_stubs()


if __name__ == "__main__":
    main()
