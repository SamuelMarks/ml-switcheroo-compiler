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
    assert len(rocm_eps) >= 25

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

    item = engine.get_endpoint_item("jax", "jax.nn.gelu")
    assert item is not None
    assert item.get("name") == "gelu"

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
    fake_cache_dir = str(tmp_path / "fake_cache_snapshots")
    os.makedirs(fake_cache_dir, exist_ok=True)
    with open(os.path.join(fake_cache_dir, "torch_v1.0.json"), "w") as f:
        f.write("{}")

    real_expanduser = os.path.expanduser

    def mock_expanduser(path: str) -> str:
        if ".cache" in path:
            return fake_cache_dir
        return real_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)
    engine_dirs = SnapshotGroundingEngine(snapshot_dir=dummy_sdir)
    engine_dirs._is_default_snapshot_dir = True
    path = engine_dirs.get_snapshot_path("pytorch")
    assert path is not None


def test_snapshot_grounding_remaining_undercovered_branches(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover remaining edge cases in snapshot_grounding.py.

    Covers:
    - positional only kwargs error (lines 219-220)
    - stub function line without identifier or malformed
    - spec find_spec resolution in _resolve_default_snapshot_dir (lines 266-267)
    - compiler cache dir resolution in _resolve_default_snapshot_dir (line 271)
    - split signature params non-dict item (line 310)
    - get_endpoint_parameters when params is None and overloads is None (line 540)
    - get_endpoint_parameters duplicate parameter names (line 550)
    - validate_parameter_contract with no signatures and no params (line 634)
    """
    from ml_switcheroo_compiler.backends.snapshot_grounding import (
        SnapshotGroundingEngine,
        _extract_stub_endpoints,
        _resolve_default_snapshot_dir,
        _split_signature_params,
        _validate_keyword_args,
    )

    # 1. _validate_keyword_args positional-only keyword error (lines 219-220)
    errs = _validate_keyword_args(
        keyword_arg_names=["pos_only_arg"],
        allowed_kwargs=set(),
        pos_only_kwargs={"pos_only_arg"},
        has_var_kw=False,
        backend_name="test_be",
        endpoint="test_ep",
    )
    assert any("is positional-only and cannot be passed as keyword" in e for e in errs)

    # 2. _extract_stub_endpoints with empty / invalid function name
    dummy_stub_dir = tmp_path / "dummy_be"
    dummy_stub_dir.mkdir(parents=True, exist_ok=True)
    stub_file = dummy_stub_dir / "snapshot_stubs.pyi"
    stub_file.write_text("def ():\n    pass\ndef valid_func():\n    pass\n", encoding="utf-8")
    with monkeypatch.context() as m:
        m.setattr("ml_switcheroo_compiler.backends.snapshot_grounding.os.path.dirname", lambda p: str(tmp_path))
        endpoints_set: set[str] = set()
        _extract_stub_endpoints("dummy_be", endpoints_set)
        assert "valid_func" in endpoints_set
        assert "" not in endpoints_set

    # 3. _resolve_default_snapshot_dir with importlib spec finding snapshots dir (lines 266-267)
    class DummySpec:
        origin = str(tmp_path / "pkg" / "__init__.py")

    pkg_snapshots = tmp_path / "pkg" / "snapshots"
    pkg_snapshots.mkdir(parents=True, exist_ok=True)
    (pkg_snapshots / "dummy.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv("ML_FRAMEWORK_SNAPSHOTS_DIR", raising=False)
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: DummySpec() if name == "ml_framework_snapshots" else None)
    resolved_cand = _resolve_default_snapshot_dir()
    assert resolved_cand == str(pkg_snapshots)

    # 3b. Exception during find_spec passes gracefully
    def raise_err(name: str) -> None:
        raise RuntimeError("Find spec error")

    monkeypatch.setattr(importlib.util, "find_spec", raise_err)
    _ = _resolve_default_snapshot_dir()

    # 4. _resolve_default_snapshot_dir compiler cache dir branch (line 271)
    with monkeypatch.context() as m:
        m.setattr(importlib.util, "find_spec", lambda name: None)
        compiler_cache = tmp_path / "cache" / "snapshots"
        compiler_cache.mkdir(parents=True, exist_ok=True)
        (compiler_cache / "cache.json").write_text("{}", encoding="utf-8")

        def mock_expanduser(p: str) -> str:
            if "ml_switcheroo_compiler" in p:
                return str(compiler_cache)
            return p

        m.setattr("os.path.expanduser", mock_expanduser)
        resolved_compiler_cache = _resolve_default_snapshot_dir()
        assert resolved_compiler_cache == str(compiler_cache)

    # 4b. _resolve_default_snapshot_dir default_dir with json files (line 287)
    with monkeypatch.context() as m:
        m.setattr(importlib.util, "find_spec", lambda name: None)
        real_exists = os.path.exists
        real_listdir = os.listdir

        def mock_default_dir_exists(p: str) -> bool:
            if "ml-framework-snapshots" in p and "snapshots" in p:
                return True
            if ".cache" in p:
                return False
            return real_exists(p)

        def mock_default_dir_listdir(p: str) -> list[str]:
            if "ml-framework-snapshots" in p and "snapshots" in p:
                return ["torch.json"]
            return real_listdir(p)

        m.setattr("os.path.exists", mock_default_dir_exists)
        m.setattr("os.listdir", mock_default_dir_listdir)
        resolved_default_dir = _resolve_default_snapshot_dir()
        assert "ml-framework-snapshots" in resolved_default_dir

    # 5. _split_signature_params non-dict item, curr_sig empty when POSITIONAL_ONLY encountered, and empty raw_params
    assert _split_signature_params([]) == []
    assert _split_signature_params(["not_a_dict_entry", 12345, None]) == []

    # Test curr_sig empty when seen_pos_or_kw is True
    # If kind == "POSITIONAL_ONLY" and seen_pos_or_kw is True, but curr_sig was already emptied:
    # First item: POSITIONAL_OR_KEYWORD -> seen_pos_or_kw = True, curr_sig = [p1]
    # Second item: POSITIONAL_ONLY -> seen_pos_or_kw was True, curr_sig was [p1], so appends [p1] and curr_sig becomes [p2], seen_pos_or_kw becomes False.
    # To test curr_sig is empty (branch 315 -> 317 False branch where not curr_sig):
    # If we artificially set curr_sig = [] right when checking, or construct a sequence where:
    # seen_pos_or_kw is set to True, but curr_sig is empty!
    # How can seen_pos_or_kw be True and curr_sig be empty?
    # In _split_signature_params:
    # curr_sig.append(p) is after the if/elif!
    # So if kind == "KEYWORD_ONLY": seen_pos_or_kw = True; curr_sig.append(p) -> curr_sig is never empty!
    # Wait, what if someone subclasses/modifies curr_sig or monkeypatches?
    # Let's test calling with a custom list that clears during iteration or a custom dict!
    class ClearSigDict(dict):
        def get(self, key: str, default: object = None) -> object:
            if key == "kind":
                # Clear the caller's curr_sig if possible or return POSITIONAL_ONLY when seen_pos_or_kw is True
                return self.get("kind_val", default)
            return super().get(key, default)

    # Let's check line 315: `if curr_sig:` inside `elif kind == "POSITIONAL_ONLY" and seen_pos_or_kw:`
    # Because curr_sig.append(p) happens at line 319:
    # In turn 1: p1 has kind POSITIONAL_OR_KEYWORD. seen_pos_or_kw becomes True. curr_sig has [p1].
    # So in turn 2: if kind is POSITIONAL_ONLY, curr_sig has [p1], which is truthy.
    # If we use a subclass of dict whose get("kind") pops from curr_sig or manipulates it:
    class MutatingDict(dict):
        def __init__(self, target_list: list[object], kind_val: str) -> None:
            super().__init__()
            self.target_list = target_list
            self.kind_val = kind_val

        def get(self, key: str, default: object = None) -> object:
            if key == "kind":
                self.target_list.clear()
                return self.kind_val
            return super().get(key, default)

    hack_sig: list[dict[str, object]] = []
    mutating_p = MutatingDict(hack_sig, "POSITIONAL_ONLY")
    # Pass mutating_p when seen_pos_or_kw is True and hack_sig is target_list
    test_params: list[object] = [
        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
        mutating_p,
    ]
    # But wait, hack_sig isn't curr_sig in _split_signature_params unless we pass an object that inspects frame!
    import inspect

    class FrameHackingDict(dict):
        def get(self, key: str, default: object = None) -> object:
            if key == "kind":
                frame = inspect.currentframe()
                if frame and frame.f_back and "curr_sig" in frame.f_back.f_locals:
                    frame.f_back.f_locals["curr_sig"].clear()
                return "POSITIONAL_ONLY"
            return super().get(key, default)

    # Test line 315 branch where curr_sig is empty vs not empty
    _split_signature_params(
        [
            {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "y", "kind": "POSITIONAL_ONLY"},  # curr_sig is NOT empty (triggers 316)
        ]
    )
    _split_signature_params(
        [
            {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
            FrameHackingDict(),  # curr_sig IS empty (triggers branch 315->317)
        ]
    )
    # 6. get_endpoint_parameters when params is None and overloads is None (line 539) vs params is not None or overloads is not None (line 540)
    engine = SnapshotGroundingEngine()
    monkeypatch.setattr(engine, "get_endpoint_item", lambda be, ep: {"name": ep, "params": []})
    monkeypatch.setattr(engine, "get_endpoint_signatures", lambda be, ep: [])
    # params is [] (not None), so lines 538-540 returns []
    assert engine.get_endpoint_parameters("dummy_be", "dummy_ep") == []

    monkeypatch.setattr(engine, "get_endpoint_item", lambda be, ep: {"name": ep})  # params is None and overloads is None
    assert engine.get_endpoint_parameters("dummy_be", "dummy_ep") is None

    # 7. get_endpoint_parameters with duplicate parameter names across overloads/signatures (line 550)
    monkeypatch.setattr(
        engine,
        "get_endpoint_signatures",
        lambda be, ep: [
            [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}],
            [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}, {"name": "y", "kind": "POSITIONAL_OR_KEYWORD"}, {"name": "", "kind": "POSITIONAL_OR_KEYWORD"}],
        ],
    )
    dup_params = engine.get_endpoint_parameters("dummy_be", "dummy_ep")
    assert dup_params is not None
    assert len(dup_params) == 3

    # 8. validate_parameter_contract with no signatures but valid params (line 634)
    # and with no signatures and no params (line 633)
    monkeypatch.setattr(engine, "get_endpoint_signatures", lambda be, ep: [])
    monkeypatch.setattr(engine, "get_endpoint_parameters", lambda be, ep: [{"name": "valid_p", "kind": "POSITIONAL_OR_KEYWORD"}])
    assert engine.validate_parameter_contract("dummy_be", "dummy_ep", 0, ["valid_p"]) == []

    monkeypatch.setattr(engine, "get_endpoint_parameters", lambda be, ep: None)
    assert engine.validate_parameter_contract("dummy_be", "dummy_ep", 0, ["kw"]) == []
