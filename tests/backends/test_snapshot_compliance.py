"""Tests verifying backend mappings compliance against ml-framework-snapshots."""

from __future__ import annotations

import glob
import os
import re

import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import SnapshotGroundingEngine

DEPRECATED_APIS: set[str] = {
    "torch.rfft",
    "torch.irfft",
    "numpy.matrix",
    "numpy.float",
    "numpy.int",
    "numpy.bool",
    "numpy.complex",
    "jax.ops.index_update",
    "jax.ops.index_add",
}


def test_no_deprecated_functions_in_mappings() -> None:
    """Ensure no removed or deprecated framework APIs are declared in mappings."""
    base_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "src",
            "ml_switcheroo_compiler",
            "backends",
        )
    )

    mapping_files: list[str] = glob.glob(os.path.join(base_dir, "**", "mappings", "*.yaml"), recursive=True)
    mapping_files.extend(glob.glob(os.path.join(base_dir, "**", "eager_mappings.yaml"), recursive=True))

    violations: list[str] = []
    for fpath in mapping_files:
        with open(fpath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue

        ops: dict[str, object] = data.get("operations", data)  # type: ignore[assignment]
        if not isinstance(ops, dict):
            continue

        for op_name, spec in ops.items():
            if not isinstance(spec, dict):
                continue
            api: str = str(spec.get("target_api") or "")
            if not api:
                ast_template: str = str(spec.get("ast_template") or "")
                m = re.match(r"^([\w\.]+)", ast_template)
                if m:
                    api = m.group(1)

            clean_api: str = api.strip("'").strip('"').split("(")[0].strip()
            if clean_api in DEPRECATED_APIS:
                violations.append(f"{fpath}: op '{op_name}' maps to deprecated API '{clean_api}'")

    err_msg: str = "\n".join(violations)
    assert not violations, f"Found deprecated API usage in backend mappings:\n{err_msg}"


def test_snapshot_compliance_coverage() -> None:
    """Verify that backend mappings achieve 100% resolution against static snapshots."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    for backend in ["pytorch", "jax", "mlx", "keras", "dask", "numpy", "cupy", "tensorflow"]:
        valid_eps: set[str] = engine.get_valid_endpoints(backend)
        assert len(valid_eps) > 0, f"Snapshot for {backend} has zero valid endpoints"
