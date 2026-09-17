"""Tests for the SnapshotGroundingEngine and backend snapshot target models."""

from __future__ import annotations

import os

import pytest

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    BackendSnapshotTargetModel,
    BackendSnapshotTargetsConfig,
    GroundingValidationError,
    SnapshotGroundingEngine,
    _extract_category_endpoints,
    _extract_item_endpoints,
    load_backend_snapshot_targets,
)


def test_load_backend_snapshot_targets_default() -> None:
    """Test loading the default backend snapshot targets configuration."""
    cfg: BackendSnapshotTargetsConfig = load_backend_snapshot_targets()
    assert cfg.version == "1.0.0"
    assert "pytorch" in cfg.targets
    assert "jax" in cfg.targets
    assert "numpy" in cfg.targets
    assert cfg.targets["pytorch"].framework == "torch"
    assert "torch" in cfg.targets["pytorch"].canonical_roots


def test_load_backend_snapshot_targets_not_found(tmp_path: object) -> None:
    """Test loading a non-existent configuration file raises FileNotFoundError.

    Args:
        tmp_path (object): Pytest temporary path fixture.
    """
    missing_path: str = os.path.join(str(tmp_path), "non_existent.yaml")
    with pytest.raises(FileNotFoundError):
        load_backend_snapshot_targets(missing_path)


def test_extract_item_endpoints() -> None:
    """Test extracting endpoints from a snapshot item dictionary."""
    endpoints: set[str] = set()
    item: dict[str, object] = {
        "api_path": "torch.add",
        "name": "add",
        "aliases": ["torch.Tensor.add", 123, ""],
    }
    _extract_item_endpoints(item, endpoints)
    assert "torch.add" in endpoints
    assert "add" in endpoints
    assert "torch.Tensor.add" in endpoints

    # Item with empty or non-string values
    item_empty: dict[str, object] = {
        "api_path": "",
        "name": None,
        "aliases": "not_a_list",
    }
    _extract_item_endpoints(item_empty, endpoints)


def test_extract_category_endpoints() -> None:
    """Test extracting endpoints from categorized snapshot structures."""
    endpoints: set[str] = set()
    _extract_category_endpoints("not_a_dict", endpoints)
    assert len(endpoints) == 0

    categories: dict[str, object] = {
        "math": [
            {"api_path": "numpy.sin", "name": "sin"},
            "invalid_item",
        ],
        "other": "not_a_list",
    }
    _extract_category_endpoints(categories, endpoints)
    assert "numpy.sin" in endpoints
    assert "sin" in endpoints


def test_snapshot_grounding_engine_endpoints() -> None:
    """Test SnapshotGroundingEngine with real framework snapshots."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    assert engine.get_snapshot_path("unknown_backend") is None

    # Test pytorch
    assert engine.is_endpoint_valid("pytorch", "torch.add")
    assert engine.is_endpoint_valid("pytorch", "torch.matmul")
    assert not engine.is_endpoint_valid("pytorch", "torch.completely_fake_endpoint_12345")

    # Test numpy
    assert engine.is_endpoint_valid("numpy", "numpy.exp")
    assert engine.is_endpoint_valid("numpy", "np.exp")
    assert not engine.is_endpoint_valid("numpy", "numpy.completely_fake_endpoint_12345")

    # Test caching
    data1: dict[str, object] = engine.load_snapshot("numpy")
    data2: dict[str, object] = engine.load_snapshot("numpy")
    assert data1 is data2
    eps1: set[str] = engine.get_valid_endpoints("numpy")
    eps2: set[str] = engine.get_valid_endpoints("numpy")
    assert eps1 is eps2


def test_snapshot_grounding_engine_env_and_custom_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test snapshot directory configuration via environment variable and explicit argument.

    Args:
        tmp_path (object): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Monkeypatch fixture.
    """
    custom_dir: str = str(tmp_path)
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(snapshot_dir=custom_dir)
    assert engine.snapshot_dir == custom_dir

    # Test via environment variable
    monkeypatch.setenv("ML_FRAMEWORK_SNAPSHOTS_DIR", custom_dir)
    engine_env: SnapshotGroundingEngine = SnapshotGroundingEngine()
    assert engine_env.snapshot_dir == custom_dir


def test_snapshot_grounding_engine_canonical_prefix() -> None:
    """Test prefix variations when only canonical path is in valid_set."""
    cfg: BackendSnapshotTargetsConfig = BackendSnapshotTargetsConfig(
        version="1.0.0",
        targets={
            "mock_backend": BackendSnapshotTargetModel(
                framework="mock_fw",
                snapshot_glob="mock_v*.json",
                canonical_roots=["mock_root"],
            )
        },
    )
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(config=cfg)
    # Mock valid endpoints to contain only 'mock_fw.submodule.func' (not 'submodule.func')
    engine._endpoint_cache["mock_backend"] = {"mock_fw.submodule.func"}

    assert engine.is_endpoint_valid("mock_backend", "mock_root.submodule.func")
    assert not engine.is_endpoint_valid("mock_backend", "mock_root.other.func")
    assert not engine.is_endpoint_valid("mock_backend", "other_root.submodule.func")


def test_snapshot_grounding_engine_missing_snapshot(tmp_path: object) -> None:
    """Test SnapshotGroundingEngine behavior when snapshot files are missing.

    Args:
        tmp_path (object): Pytest temporary path fixture.
    """
    empty_dir: str = str(tmp_path)
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(snapshot_dir=empty_dir)

    assert engine.get_snapshot_path("pytorch") is None
    with pytest.raises(FileNotFoundError):
        engine.load_snapshot("pytorch")

    assert engine.get_valid_endpoints("pytorch") == set()
    assert not engine.is_endpoint_valid("pytorch", "torch.add")
    assert not engine.is_endpoint_valid("unknown_backend", "dummy")


def test_snapshot_grounding_engine_corrupted_config(tmp_path: object) -> None:
    """Test SnapshotGroundingEngine when a target has no configured target model.

    Args:
        tmp_path (object): Pytest temporary path fixture.
    """
    cfg: BackendSnapshotTargetsConfig = BackendSnapshotTargetsConfig(version="1.0.0", targets={})
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg)
    assert not engine.is_endpoint_valid("pytorch", "torch.add")
    with pytest.raises(FileNotFoundError):
        engine.load_snapshot("pytorch")


def test_backend_snapshot_targets_hardware_and_edge() -> None:
    """Test that hardware and edge snapshot targets are configured in targets schema."""
    cfg: BackendSnapshotTargetsConfig = load_backend_snapshot_targets()
    for target_key in ("edge_onnx", "webgpu", "metal", "rocm", "cuda", "edge_stablehlo"):
        assert target_key in cfg.targets
        target = cfg.targets[target_key]
        assert len(target.canonical_roots) > 0
        assert target.snapshot_glob.endswith(".json")


def test_snapshot_grounding_engine_parameters_and_return_type() -> None:
    """Test retrieving parameter lists and return type metadata from snapshots."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # Unknown backend or non-existent endpoint
    assert engine.get_endpoint_parameters("unknown_backend", "fake_op") is None
    assert engine.get_endpoint_return_type("unknown_backend", "fake_op") is None
    assert engine.get_endpoint_parameters("numpy", "completely_fake_endpoint_999") is None
    assert engine.get_endpoint_return_type("numpy", "completely_fake_endpoint_999") is None

    # Real numpy endpoint (numpy.exp)
    params = engine.get_endpoint_parameters("numpy", "numpy.exp")
    assert params is not None
    assert isinstance(params, list)
    assert any(p.get("name") in ("x", "args") for p in params)

    ret_type = engine.get_endpoint_return_type("numpy", "numpy.exp")
    assert ret_type is not None
    assert "ndarray" in ret_type or "scalar" in ret_type

    # Test via canonical root / alias
    item = engine.get_endpoint_item("numpy", "np.exp")
    assert item is not None
    assert item.get("name") == "exp"


def test_snapshot_grounding_hardware_and_mlir() -> None:
    """Test grounding for hardware (CUDA, ROCm) and edge MLIR snapshots."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # MLIR operations
    assert engine.is_endpoint_valid("edge_mlir", "ForOp") or len(engine.get_valid_endpoints("edge_mlir")) > 50

    # CUDA SASS instructions
    cuda_eps: set[str] = engine.get_valid_endpoints("cuda")
    assert len(cuda_eps) > 100

    # ROCm RDNA instructions
    rocm_eps: set[str] = engine.get_valid_endpoints("rocm")
    assert len(rocm_eps) > 100

    # Missing upstream targets
    for missing in ("edge_onnx", "webgpu", "metal"):
        target = engine.config.targets[missing]
        assert target.status == "missing_upstream"
        assert target.fallback_schema == "synthetic"
        assert engine.get_snapshot_path(missing) is None


def test_snapshot_grounding_keyword_validation_discrepancies() -> None:
    """Test _validate_keyword_args discrepancy pairs, empty kwargs, and unrecognized kwargs."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # Empty kw, matching kw, leading underscore kw
    errors: list[str] = engine.validate_parameter_contract("pytorch", "torch.matmul", 2, ["", "out", "_private_kw"])
    assert not errors

    # Discrepancy pairs: 'axis' provided where 'dim' is expected
    errors = engine.validate_parameter_contract("pytorch", "torch.sum", 1, ["axis"])
    assert any("does not accept keyword 'axis'. Did you mean 'dim'?" in e for e in errors)

    # Discrepancy pairs: 'keepdims' vs 'keepdim'
    errors = engine.validate_parameter_contract("pytorch", "torch.sum", 1, ["keepdims"])
    assert any("does not accept keyword 'keepdims'. Did you mean 'keepdim'?" in e for e in errors)

    # Discrepancy pairs: 'split_size_or_sections' vs 'indices_or_sections'
    errors = engine.validate_parameter_contract("numpy", "numpy.split", 1, ["split_size_or_sections"])
    assert any("split_size_or_sections" in e and "indices_or_sections" in e for e in errors)

    # Unknown keyword argument
    errors = engine.validate_parameter_contract("pytorch", "torch.matmul", 2, ["completely_unknown_kwarg"])
    assert any("Unknown keyword argument 'completely_unknown_kwarg'" in e for e in errors)

    # Var keyword endpoint accepts arbitrary keywords without error
    custom_cfg = BackendSnapshotTargetsConfig(
        targets={
            "custom": BackendSnapshotTargetModel(
                framework="custom",
                canonical_roots=["custom"],
                snapshot_glob="*.json",
            )
        }
    )
    custom_engine = SnapshotGroundingEngine(snapshot_dir="/nonexistent", config=custom_cfg)
    custom_engine._endpoint_cache["custom"] = {"custom.var_fn"}
    custom_engine._endpoint_item_cache["custom"] = {
        "custom.var_fn": {
            "name": "var_fn",
            "params": [
                {"name": "args", "kind": "VAR_POSITIONAL"},
                {"name": "kwargs", "kind": "VAR_KEYWORD"},
            ],
        }
    }
    var_kw_errors = custom_engine.validate_parameter_contract("custom", "custom.var_fn", 10, ["any_kwarg"])
    assert not var_kw_errors


def test_snapshot_grounding_framework_file_candidate(tmp_path: object) -> None:
    """Test resolving snapshot path via target.framework_file candidate."""
    import os

    base_dir = str(tmp_path)
    snap_dir = os.path.join(base_dir, "snapshots")
    fw_dir = os.path.join(base_dir, "frameworks")
    os.makedirs(fw_dir)
    fw_file = os.path.join(fw_dir, "nvidia_ptx_exhaustive.json")
    with open(fw_file, "w") as f:
        f.write("{}")

    engine = SnapshotGroundingEngine(snapshot_dir=snap_dir)
    resolved = engine.get_snapshot_path("cuda")
    assert resolved == fw_file


def test_snapshot_grounding_list_snapshot_data(tmp_path: object) -> None:
    """Test loading snapshot formatted as a list of items rather than a category dictionary."""
    import json
    import os

    snap_file = os.path.join(str(tmp_path), "list_snap.json")
    with open(snap_file, "w") as f:
        json.dump(
            [
                "not_a_dict",
                {
                    "name": "list_op",
                    "api_path": "list_framework.list_op",
                    "aliases": ["list_framework.op_alias"],
                    "params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
                },
            ],
            f,
        )

    cfg = BackendSnapshotTargetsConfig(
        targets={
            "list_framework": BackendSnapshotTargetModel(
                framework="list_framework",
                canonical_roots=["list_framework"],
                snapshot_glob="list_snap.json",
            )
        }
    )
    engine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg)
    valid_eps = engine.get_valid_endpoints("list_framework")
    assert "list_framework.list_op" in valid_eps
    assert "list_framework.op_alias" in valid_eps


def test_snapshot_grounding_endpoint_item_candidate_roots(tmp_path: object) -> None:
    """Test resolving endpoint item across canonical roots and alternative roots."""
    engine = SnapshotGroundingEngine()

    item = engine.get_endpoint_item("jax", "jax.scipy.linalg.lu")
    assert item is not None
    assert item.get("name") == "lu"

    sin_item = engine.get_endpoint_item("jax", "jnp.sin")
    assert sin_item is not None

    missing_item = engine.get_endpoint_item("jax", "jax.definitely_nonexistent_op_12345")
    assert missing_item is None

    assert engine.get_endpoint_item("unknown_backend", "any_op") is None

    # Synthetic test to guarantee alt_root, suffix, and target.framework fallback branches
    import json
    import os

    snap_file = os.path.join(str(tmp_path), "roots_test.json")
    with open(snap_file, "w") as f:
        json.dump(
            {
                "categories": {
                    "ops": [
                        {"name": "bare_op", "api_path": "bare_op"},
                        {"name": "alt_op", "api_path": "rootB.alt_op"},
                        {"name": "canon_op", "api_path": "base_framework.canon_op"},
                    ]
                }
            },
            f,
        )

    cfg = BackendSnapshotTargetsConfig(
        targets={
            "mock_fw": BackendSnapshotTargetModel(
                framework="base_framework",
                canonical_roots=["rootA", "rootB"],
                snapshot_glob="roots_test.json",
            )
        }
    )
    mock_engine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg)
    # Populate cache directly with multi-segment paths so suffix isn't directly in item_map
    mock_engine._endpoint_cache["mock_fw"] = {"rootB.sub.alt_op", "base_framework.sub.canon_op"}
    mock_engine._endpoint_item_cache["mock_fw"] = {
        "rootB.sub.alt_op": {"name": "alt_op"},
        "base_framework.sub.canon_op": {"name": "canon_op"},
    }
    # rootA.sub.alt_op -> suffix 'sub.alt_op' not in map, alt_root rootB -> rootB.sub.alt_op (line 335 hit)
    assert mock_engine.get_endpoint_item("mock_fw", "rootA.sub.alt_op") is not None
    # rootA.sub.canon_op -> suffix 'sub.canon_op' not in map, alt_roots not in map, target.framework (base_framework.sub.canon_op) hit (line 338 hit)
    assert mock_engine.get_endpoint_item("mock_fw", "rootA.sub.canon_op") is not None


def test_snapshot_grounding_nonversioned_snapshot_glob(tmp_path: object) -> None:
    """Test get_snapshot_path candidate selection when candidate snapshot file has no version."""
    import os

    snap_dir = os.path.join(str(tmp_path), "snapshots")
    os.makedirs(snap_dir)
    fw_dir = os.path.join(str(tmp_path), "frameworks")
    os.makedirs(fw_dir)
    unversioned_file = os.path.join(snap_dir, "custom_unversioned.json")
    with open(unversioned_file, "w") as f:
        f.write("{}")

    cfg = BackendSnapshotTargetsConfig(
        targets={
            "custom": BackendSnapshotTargetModel(
                framework="custom",
                canonical_roots=["custom"],
                snapshot_glob="custom_*.json",
                framework_file="missing_framework_file.json",
            ),
            "custom_none": BackendSnapshotTargetModel(
                framework="custom_none",
                canonical_roots=["custom_none"],
                snapshot_glob="nonexistent_*.json",
                framework_file="missing_fw.json",
            ),
        }
    )
    engine = SnapshotGroundingEngine(snapshot_dir=snap_dir, config=cfg)
    assert engine.get_snapshot_path("custom") == unversioned_file
    assert engine.get_snapshot_path("custom_none") is None


def test_snapshot_grounding_unrecognized_snapshot_structure(tmp_path: object) -> None:
    """Test get_valid_endpoints when snapshot JSON is neither dict nor list."""
    import json
    import os

    snap_file = os.path.join(str(tmp_path), "invalid_shape.json")
    with open(snap_file, "w") as f:
        json.dump("not_a_dict_or_list", f)

    cfg = BackendSnapshotTargetsConfig(
        targets={
            "invalid_fw": BackendSnapshotTargetModel(
                framework="invalid_fw",
                canonical_roots=["invalid_fw"],
                snapshot_glob="invalid_shape.json",
            )
        }
    )
    engine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg)
    assert engine.get_valid_endpoints("invalid_fw") == set()


def test_snapshot_grounding_overloads_and_empty_returns(tmp_path: object) -> None:
    """Test endpoints with overloads parameter specifications and empty/unannotated returns_type."""
    engine = SnapshotGroundingEngine()

    params = engine.get_endpoint_parameters("numpy", "numpy.all")
    assert params is not None
    assert len(params) >= 2

    assert engine.get_endpoint_return_type("pytorch", "torch.Tensor.median") is None

    import json
    import os

    snap_file = os.path.join(str(tmp_path), "dummy_overloads.json")
    with open(snap_file, "w") as f:
        json.dump(
            {
                "categories": {
                    "ops": [
                        {
                            "name": "no_params_op",
                            "api_path": "test_fw.no_params_op",
                            "params": None,
                            "overloads": None,
                        },
                        {
                            "name": "bad_overload_op",
                            "api_path": "test_fw.bad_overload_op",
                            "params": None,
                            "overloads": ["not_a_dict", {"params": [{"name": "x"}]}],
                        },
                    ]
                }
            },
            f,
        )

    cfg = BackendSnapshotTargetsConfig(
        targets={
            "test_fw": BackendSnapshotTargetModel(
                framework="test_fw",
                canonical_roots=["test_fw"],
                snapshot_glob="dummy_overloads.json",
            )
        }
    )
    dummy_engine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg)
    assert dummy_engine.get_endpoint_parameters("test_fw", "test_fw.no_params_op") is None
    ov_params = dummy_engine.get_endpoint_parameters("test_fw", "test_fw.bad_overload_op")
    assert ov_params is not None
    assert len(ov_params) == 1
    assert ov_params[0]["name"] == "x"


def test_snapshot_grounding_parameter_contract_validation_branches() -> None:
    """Test missing required arguments, raise_on_error flag, and valid contract checks."""
    engine = SnapshotGroundingEngine()

    missing_errors = engine.validate_parameter_contract("numpy", "dot", 0, [])
    assert len(missing_errors) == 1
    assert "Missing required arguments" in missing_errors[0]

    # Too many positional arguments (dot takes at most 9 pos params, we give 10)
    too_many_errors = engine.validate_parameter_contract("numpy", "dot", 10, [])
    assert len(too_many_errors) == 1
    assert "Too many positional arguments" in too_many_errors[0]

    # Required positional argument satisfied with 2 positional arguments
    assert not engine.validate_parameter_contract("numpy", "dot", 2, [])

    supplied_via_kw = engine.validate_parameter_contract("numpy", "dot", 0, ["a", "b"])
    assert not supplied_via_kw

    with pytest.raises(GroundingValidationError) as exc:
        engine.validate_parameter_contract("numpy", "dot", 0, [], raise_on_error=True)
    assert "Missing required arguments" in str(exc.value)

    assert engine.validate_parameter_contract("numpy", "nonexistent_op", 5, ["foo"]) == []


def test_snapshot_grounding_is_endpoint_valid_branches() -> None:
    """Test is_endpoint_valid for unknown backends, empty snapshots, direct match, and suffix matching."""
    engine = SnapshotGroundingEngine()

    assert not engine.is_endpoint_valid("completely_unknown_backend", "torch.add")
    assert not engine.is_endpoint_valid("edge_onnx", "onnx.Add")
    assert engine.is_endpoint_valid("pytorch", "torch.add")
    assert engine.is_endpoint_valid("jax", "jnp.sin")
    assert engine.is_endpoint_valid("jax", "jax.sin")
    assert not engine.is_endpoint_valid("jax", "jax.totally_invalid_function_name_999")


def test_snapshot_grounding_cache_resolution_and_alt_roots(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolution fallback branches for compiler cache, user cache, and alternative root lookups."""
    import json

    # 1. Test alt_root lookup in get_endpoint_item
    item_file = os.path.join(str(tmp_path), "custom_alt_roots.json")
    with open(item_file, "w") as f:
        json.dump(
            {
                "categories": {
                    "ops": [
                        {
                            "name": "alt_op",
                            "api_path": "cp.alt_op",
                            "params": [],
                        }
                    ]
                }
            },
            f,
        )

    cfg = BackendSnapshotTargetsConfig(
        targets={
            "custom_fw": BackendSnapshotTargetModel(
                framework="cupy",
                canonical_roots=["cupy", "cp"],
                snapshot_glob="custom_alt_roots.json",
            )
        }
    )
    alt_engine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg)
    found_item = alt_engine.get_endpoint_item("custom_fw", "cupy.alt_op")
    assert found_item is not None
    assert found_item.get("api_path") == "cp.alt_op"

    # 2. Test directory resolution branches when compiler cache does not exist
    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_DIR", raising=False)

    fake_default = str(tmp_path / "default_dir")
    fake_compiler = str(tmp_path / "compiler_dir")
    fake_cache = str(tmp_path / "cache_dir")
    os.makedirs(fake_default, exist_ok=True)
    os.makedirs(fake_cache, exist_ok=True)

    # Sub-case A: default_has_jsons is True
    with open(os.path.join(fake_default, "dummy.json"), "w") as f:
        f.write("{}")

    real_exists = os.path.exists
    real_listdir = os.listdir

    def mock_exists(p: str) -> bool:
        if "ml_switcheroo_compiler" in p and "snapshots" in p:
            return False
        if "ml_framework_snapshots" in p and "snapshots" in p:
            return True
        return real_exists(p)

    def mock_listdir(p: str) -> list[str]:
        if "ml_framework_snapshots" in p and "snapshots" in p:
            return ["torch_v1.0.json"]
        return real_listdir(p)

    monkeypatch.setattr(os.path, "exists", mock_exists)
    monkeypatch.setattr(os, "listdir", mock_listdir)

    eng_default = SnapshotGroundingEngine()
    assert "ml_framework_snapshots" in eng_default.snapshot_dir

    # Sub-case B: default_has_jsons is False, cache has jsons
    def mock_listdir_cache(p: str) -> list[str]:
        if ".cache" in p and "ml_framework_snapshots" in p:
            return ["torch_v1.0.json"]
        return []

    monkeypatch.setattr(os, "listdir", mock_listdir_cache)
    eng_cache = SnapshotGroundingEngine()
    assert ".cache" in eng_cache.snapshot_dir

    # Sub-case C: neither has jsons
    monkeypatch.setattr(os, "listdir", lambda p: [])
    eng_empty = SnapshotGroundingEngine()
    assert "ml_framework_snapshots" in eng_empty.snapshot_dir

    # Sub-case D: line 270 cd not in dirs_to_check in get_snapshot_path
    monkeypatch.undo()
    dummy_sdir = str(tmp_path / "explicit_default")
    os.makedirs(dummy_sdir, exist_ok=True)
    engine_dirs = SnapshotGroundingEngine(snapshot_dir=dummy_sdir)
    engine_dirs._is_default_snapshot_dir = True
    path = engine_dirs.get_snapshot_path("pytorch")
    assert path is not None
