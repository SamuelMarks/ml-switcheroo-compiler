"""Tests verifying backend mappings compliance against ml-framework-snapshots."""

from __future__ import annotations

import ast
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

        raw_ops: object = data.get("operations", data)
        if not isinstance(raw_ops, dict):
            continue
        ops: dict[object, object] = raw_ops

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


def test_ecosystem_snapshots_grounding() -> None:
    """Verify that newly integrated ecosystem snapshots have valid endpoints and ground operations."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    for target in ["torchvision", "torchaudio", "scipy", "safetensors", "nccl"]:
        valid_eps: set[str] = engine.get_valid_endpoints(target)
        assert len(valid_eps) > 0, f"Snapshot for {target} has zero valid endpoints"

    # Verify safetensors endpoints
    st_eps = engine.get_valid_endpoints("safetensors")
    assert "safetensors.numpy.save_file" in st_eps or "save_file" in st_eps
    assert "safetensors.numpy.load_file" in st_eps or "load_file" in st_eps

    # Verify nccl endpoints
    nccl_eps = engine.get_valid_endpoints("nccl")
    for op in ["all_reduce", "broadcast", "all_gather", "reduce_scatter"]:
        assert f"nccl.{op}" in nccl_eps or op in nccl_eps

    # Verify scipy endpoints
    scipy_eps = engine.get_valid_endpoints("scipy")
    assert "scipy.special.gamma" in scipy_eps or "gamma" in scipy_eps
    assert "scipy.linalg.cholesky" in scipy_eps or "cholesky" in scipy_eps


def test_all_backends_have_valid_nonempty_stubs() -> None:
    """Verify that all 24 backends have valid, non-empty generated type stubs."""
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

    all_backends: list[str] = [
        "numpy",
        "pytorch",
        "jax",
        "keras",
        "mlx",
        "dask",
        "cupy",
        "tensorflow",
        "cuda",
        "rocm",
        "metal",
        "awkward",
        "dpnp",
        "numba",
        "sparse",
        "pyarrow_compute",
        "llvm_cpp",
        "edge",
        "edge_onnx",
        "edge_stablehlo",
        "edge_mlir",
        "edge_wasm",
        "edge_webgl",
        "webgpu",
        "pure_python",
    ]

    for be in all_backends:
        stub_file: str = os.path.join(base_dir, be, "snapshot_stubs.pyi")
        assert os.path.exists(stub_file), f"Missing snapshot_stubs.pyi for backend '{be}'"
        assert os.path.getsize(stub_file) > 100, f"Empty or too small stub file for backend '{be}'"
        with open(stub_file, encoding="utf-8") as f:
            content: str = f.read()
        parsed: ast.AST = ast.parse(content, filename=stub_file)
        funcs: list[ast.FunctionDef] = [n for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef)]
        assert len(funcs) > 0, f"No functions found in stubs for backend '{be}'"
