"""Static Grounding Engine for Backend API Mappings using ml-ecosystem-snapshots.

This module provides schema-validated utilities to verify backend mapping declarations
against static JSON and compressed JSON.GZ snapshots extracted by the
ml-ecosystem-snapshots library.
"""

from __future__ import annotations

import gzip
import json
import os

import yaml
from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef
from pydantic import BaseModel, Field


class BackendSnapshotTargetModel(BaseModel):
    """Pydantic specification for a target backend framework snapshot binding."""

    framework: str = Field(description="Canonical framework name in ml-ecosystem-snapshots.")
    snapshot_glob: str = Field(description="Glob pattern for locating JSON snapshot file.")
    status: str = Field(
        default="available",
        description="Snapshot status in ml-ecosystem-snapshots ('available' or 'missing_upstream').",
    )
    framework_file: str | None = Field(
        default=None,
        description="Optional framework specification file in ml_ecosystem_snapshots/frameworks.",
    )
    fallback_schema: str | None = Field(
        default=None,
        description="Fallback schema policy when snapshot is missing upstream.",
    )
    canonical_roots: list[str] = Field(
        default_factory=list,
        description="List of valid root module names recognized for this target backend.",
    )


class BackendSnapshotTargetsConfig(BaseModel):
    """Root configuration model for backend snapshot targets."""

    version: str = Field(default="1.0.0", description="Configuration schema version.")
    targets: dict[str, BackendSnapshotTargetModel] = Field(
        default_factory=dict,
        description="Dictionary mapping backend identifiers to snapshot targets.",
    )


def load_backend_snapshot_targets(
    config_path: str | None = None,
) -> BackendSnapshotTargetsConfig:
    """Load and validate the backend snapshot targets configuration file.

    Args:
        config_path (str | None): Optional custom file path to the YAML configuration.

    Returns:
        BackendSnapshotTargetsConfig: Validated snapshot targets configuration model.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If YAML parsing fails or validation fails.
    """
    if config_path is None:
        base_dir: str = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "backend_snapshot_targets.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Snapshot targets configuration not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw_data: dict[str, object] = yaml.safe_load(f) or {}

    return BackendSnapshotTargetsConfig.model_validate(raw_data)


def _extract_item_endpoints(
    item: dict[str, object],
    endpoints: set[str],
    item_map: dict[str, dict[str, object]] | None = None,
) -> None:
    """Extract individual API path, name, and aliases from a snapshot item.

    Args:
        item (dict[str, object]): An entity dictionary from snapshot JSON.
        endpoints (set[str]): Destination set to populate with valid endpoint strings.
        item_map (dict[str, dict[str, object]] | None): Optional destination map from endpoint to item data.
    """
    for key in ("api_path", "name", "mnemonic"):
        val: object = item.get(key)
        if isinstance(val, str) and val:
            endpoints.add(val)
            if item_map is not None:
                item_map[val] = item

    aliases_obj: object = item.get("aliases")
    if isinstance(aliases_obj, list):
        for a in aliases_obj:
            if isinstance(a, str) and a:
                endpoints.add(a)
                if item_map is not None:
                    item_map[a] = item


def _extract_category_endpoints(
    categories_obj: object,
    endpoints: set[str],
    item_map: dict[str, dict[str, object]] | None = None,
) -> None:
    """Extract endpoints from category groupings in snapshot data.

    Args:
        categories_obj (object): Categories mapping from snapshot JSON.
        endpoints (set[str]): Destination set to populate with valid endpoint strings.
        item_map (dict[str, dict[str, object]] | None): Optional destination map from endpoint to item data.
    """
    if not isinstance(categories_obj, dict):
        return
    for items_obj in categories_obj.values():
        if isinstance(items_obj, list):
            for item_obj in items_obj:
                if isinstance(item_obj, dict):
                    _extract_item_endpoints(item_obj, endpoints, item_map)


class GroundingValidationError(ValueError):
    """Exception raised when an API call fails static framework snapshot verification."""


def _load_standard_arg_map() -> dict[str, str]:
    """Load standard argument mapping from ml-ecosystem-snapshots or provide fallback.

    Returns:
        dict[str, str]: Mapping from variant argument names to standardized names.
    """
    try:
        import importlib

        mod = importlib.import_module("ml_ecosystem_snapshots.models")
        loaded: object = getattr(mod, "STANDARD_ARG_MAP", None)
        if isinstance(loaded, dict):
            return {str(k): str(v) for k, v in loaded.items()}
    except Exception:
        pass
    return {
        "x": "input",
        "inputs": "input",
        "input_tensor": "input",
        "y": "other",
        "other_tensor": "other",
        "dim": "dim",
        "axis": "dim",
        "keepdim": "keepdims",
        "keep_dims": "keepdims",
        "keepdims": "keepdims",
    }


STANDARD_ARG_MAP: dict[str, str] = _load_standard_arg_map()


DISCREPANCY_PAIRS: list[tuple[str, str]] = [
    ("dim", "axis"),
    ("axis", "dim"),
    ("dims", "axis"),
    ("axis", "dims"),
    ("dimension", "axis"),
    ("axis", "dimension"),
    ("dim", "dimension"),
    ("dimension", "dim"),
    ("keepdim", "keepdims"),
    ("keepdims", "keepdim"),
    ("input", "a"),
    ("a", "input"),
    ("input", "x"),
    ("x", "input"),
    ("a", "x"),
    ("x", "a"),
    ("input_tensor", "input"),
    ("input", "input_tensor"),
    ("input_tensor", "x"),
    ("x", "input_tensor"),
    ("input_tensor", "a"),
    ("a", "input_tensor"),
    ("tensor", "input"),
    ("input", "tensor"),
    ("other", "b"),
    ("b", "other"),
    ("other", "y"),
    ("y", "other"),
    ("b", "y"),
    ("y", "b"),
    ("split_size_or_sections", "indices_or_sections"),
    ("indices_or_sections", "split_size_or_sections"),
    ("num_or_size_splits", "indices_or_sections"),
    ("indices_or_sections", "num_or_size_splits"),
    ("num_or_size_splits", "split_size_or_sections"),
    ("split_size_or_sections", "num_or_size_splits"),
    ("ord", "p"),
    ("p", "ord"),
    ("unbiased", "ddof"),
    ("ddof", "unbiased"),
    ("correction", "ddof"),
    ("ddof", "correction"),
    ("unbiased", "correction"),
    ("correction", "unbiased"),
    ("generator", "seed"),
    ("seed", "generator"),
    ("generator", "key"),
    ("key", "generator"),
    ("seed", "key"),
    ("key", "seed"),
    ("dtype", "output_type"),
    ("output_type", "dtype"),
]


def _find_discrepancy_replacement(
    kw: str,
    allowed_kwargs: set[str],
    ep_allowed: set[str],
    backend_name: str,
    endpoint: str,
) -> str | None:
    """Find a canonical replacement for a mismatched keyword argument.

    Args:
        kw (str): Provided keyword argument name.
        allowed_kwargs (set[str]): Allowed keyword arguments in the specific signature overload.
        ep_allowed (set[str]): All allowed keyword arguments across all signature overloads.
        backend_name (str): Target framework name.
        endpoint (str): Fully qualified API endpoint.

    Returns:
        str | None: Suggested canonical argument replacement if found.
    """
    try:
        import importlib

        mcp = importlib.import_module("ml_ecosystem_snapshots.mcp_server")
        explain_fn = getattr(mcp, "explain_anti_pattern", None)
        if explain_fn is not None:
            info: dict[str, object] = explain_fn(backend_name, endpoint, kw)
            if info.get("is_known_anti_pattern"):
                cand: str = str(info.get("canonical_argument") or "")
                if cand and (cand in allowed_kwargs or cand in ep_allowed):
                    return cand
    except Exception:
        pass

    std: str | None = STANDARD_ARG_MAP.get(kw)
    if std is not None:
        for cand in sorted(allowed_kwargs | ep_allowed):
            if cand != kw and STANDARD_ARG_MAP.get(cand) == std:
                return cand

    for provided, expected in DISCREPANCY_PAIRS:
        if kw == provided and (expected in allowed_kwargs or expected in ep_allowed):
            return expected

    return None


def _validate_keyword_args(
    keyword_arg_names: list[str],
    allowed_kwargs: set[str],
    pos_only_kwargs: set[str],
    has_var_kw: bool,
    backend_name: str,
    endpoint: str,
    endpoint_allowed_kwargs: set[str] | None = None,
) -> list[str]:
    """Validate keyword arguments against allowed kwargs and check discrepancy pairs.

    Args:
        keyword_arg_names (list[str]): List of keyword argument names.
        allowed_kwargs (set[str]): Allowed keyword arguments.
        pos_only_kwargs (set[str]): Positional-only parameter names.
        has_var_kw (bool): Whether backend endpoint accepts arbitrary var keywords.
        backend_name (str): Target backend framework identifier.
        endpoint (str): Fully qualified endpoint string.
        endpoint_allowed_kwargs (set[str] | None): Full set of allowed keyword arguments across all endpoint overloads.

    Returns:
        list[str]: Validation error messages.
    """
    errors: list[str] = []
    if has_var_kw:
        return errors

    ep_allowed: set[str] = endpoint_allowed_kwargs if endpoint_allowed_kwargs is not None else allowed_kwargs

    for kw in keyword_arg_names:
        if not kw or kw in allowed_kwargs or kw.startswith("_"):
            continue
        if kw in pos_only_kwargs:
            errors.append(f"Argument '{kw}' for endpoint '{endpoint}' in backend '{backend_name}' is positional-only and cannot be passed as keyword")
            continue

        replacement: str | None = _find_discrepancy_replacement(kw, allowed_kwargs, ep_allowed, backend_name, endpoint)
        if replacement is not None:
            errors.append(f"Endpoint '{endpoint}' in backend '{backend_name}' does not accept keyword '{kw}'. Did you mean '{replacement}'?")
        else:
            errors.append(f"Unknown keyword argument '{kw}' for endpoint '{endpoint}' in backend '{backend_name}'")
    return errors


def _extract_stub_endpoints(backend_name: str, endpoints: set[str]) -> None:
    """Extract declared function endpoints from a backend snapshot_stubs.pyi file.

    Args:
        backend_name (str): Backend directory name.
        endpoints (set[str]): Destination set to populate with valid function names.
    """
    stub_path: str = os.path.join(os.path.dirname(__file__), backend_name, "snapshot_stubs.pyi")
    if os.path.exists(stub_path):
        with open(stub_path, encoding="utf-8") as f:
            for line in f:
                sline: str = line.strip()
                if sline.startswith("def "):
                    func_name: str = sline[4:].split("(")[0].strip()
                    if func_name and func_name.isidentifier():
                        endpoints.add(func_name)


def _resolve_env_or_package_snapshot_dir() -> str | None:
    """Resolve snapshot directory from environment variables or installed packages.

    Returns:
        str | None: Absolute path to the located snapshot directory, or None if not found.
    """
    for env_var in ("ML_ECOSYSTEM_SNAPSHOTS_DIR", "ML_FRAMEWORK_SNAPSHOTS_DIR"):
        env_dir: str | None = os.environ.get(env_var)
        if env_dir and os.path.exists(env_dir):
            return os.path.abspath(env_dir)

    for pkg_name in ("ml_ecosystem_snapshots", "ml_framework_snapshots"):
        try:
            import importlib.util

            spec = importlib.util.find_spec(pkg_name)
            if spec is not None and spec.origin is not None:
                cand: str = os.path.join(os.path.dirname(spec.origin), "snapshots")
                if os.path.exists(cand) and any(f.endswith(".json") or f.endswith(".json.gz") for f in os.listdir(cand)):
                    return os.path.abspath(cand)
        except Exception:
            pass

    return None


def _resolve_default_snapshot_dir() -> str:
    """Resolve default static snapshot directory across environment, packages, and caches.

    Returns:
        str: Absolute path to the located snapshot directory.
    """
    env_or_pkg: str | None = _resolve_env_or_package_snapshot_dir()
    if env_or_pkg is not None:
        return env_or_pkg

    candidate_dirs: list[str] = [
        os.path.expanduser(os.path.join("~", ".cache", "ml_ecosystem_snapshots", "snapshots")),
        os.path.expanduser(os.path.join("~", ".cache", "ml_ecosystem_snapshots")),
        os.path.expanduser(os.path.join("~", ".cache", "ml_switcheroo_compiler", "snapshots")),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "ml-ecosystem-snapshots",
                "src",
                "ml_ecosystem_snapshots",
                "snapshots",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "ml-ecosystem-snapshots",
                "src",
                "ml_framework_snapshots",
                "snapshots",
            )
        ),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "ml-framework-snapshots",
                "src",
                "ml_framework_snapshots",
                "snapshots",
            )
        ),
        os.path.expanduser(os.path.join("~", ".cache", "ml_framework_snapshots", "snapshots")),
    ]
    for cdir in candidate_dirs:
        if os.path.exists(cdir) and any(f.endswith(".json") or f.endswith(".json.gz") for f in os.listdir(cdir)):
            return cdir

    return candidate_dirs[3]


def _is_snapshot_file_complete(file_path: str, backend_name: str) -> bool:
    """Verify if candidate snapshot JSON contains required operational categories.

    Args:
        file_path (str): Candidate snapshot file path (.json or .json.gz).
        backend_name (str): Target backend name.

    Returns:
        bool: True if snapshot contains required categories or is non-dask.
    """
    if backend_name != "dask":
        return True
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt", encoding="utf-8") as gf:
                data: dict[str, object] = json.load(gf)
        else:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        categories: object = data.get("categories") if isinstance(data, dict) else None
        return isinstance(categories, dict) and "array" in categories
    except Exception:
        return False


DOCSTRING_SECTION_HEADERS: frozenset[str] = frozenset(
    {
        "Examples",
        "Notes",
        "References",
        "Parameters",
        "Returns",
        "Yields",
        "Raises",
        "See",
        "Also",
        "Attributes",
        "Methods",
        "Warns",
        "Warnings",
    }
)


def compute_levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: Edit distance between s1 and s2.
    """
    try:
        import importlib

        mod = importlib.import_module("ml_ecosystem_snapshots.grounding.engine")
        lev_fn = getattr(mod, "compute_levenshtein", None)
        if lev_fn is not None:
            return int(lev_fn(s1, s2))
    except Exception:
        pass

    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row: list[int] = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions: int = previous_row[j + 1] + 1
            deletions: int = current_row[j] + 1
            substitutions: int = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    return previous_row[-1]


def _split_signature_params(raw_params: list[object]) -> list[list[dict[str, object]]]:
    """Split concatenated or malformed parameter lists into coherent Python signatures.

    Args:
        raw_params (list[object]): Raw list of parameter dictionaries.

    Returns:
        list[list[dict[str, object]]]: Extracted distinct signature parameter lists.
    """
    signatures: list[list[dict[str, object]]] = []
    curr_sig: list[dict[str, object]] = []
    seen_pos_or_kw: bool = False
    for p in raw_params:
        if not isinstance(p, dict):
            continue
        p_name: object = p.get("name")
        if not isinstance(p_name, str) or not p_name.isidentifier() or p_name in DOCSTRING_SECTION_HEADERS:
            continue
        kind: object = p.get("kind")
        if kind in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"):
            seen_pos_or_kw = True
        elif kind == "POSITIONAL_ONLY" and seen_pos_or_kw:
            signatures.append(curr_sig)
            curr_sig = []
            seen_pos_or_kw = False
        curr_sig.append(p)
    if curr_sig:
        signatures.append(curr_sig)
    return signatures


def _hydrate_ghost_ref(data: dict[str, object]) -> ExtendedGhostRef | None:
    """Hydrate a snapshot item dictionary into an ExtendedGhostRef model if compatible.

    Args:
        data (dict[str, object]): Raw dictionary representing an API symbol.

    Returns:
        ExtendedGhostRef | None: Hydrated ExtendedGhostRef instance, or None if invalid.
    """
    clean_data: dict[str, object] = dict(data)
    if "kind" not in clean_data or not clean_data["kind"]:
        clean_data["kind"] = "function"
    if "name" not in clean_data or not clean_data["name"]:
        clean_data["name"] = str(clean_data.get("mnemonic") or clean_data.get("api_path") or "unnamed")
    if "api_path" not in clean_data or not clean_data["api_path"]:
        clean_data["api_path"] = str(clean_data["name"])
    if clean_data.get("params") is None:
        clean_data["params"] = []
    if clean_data.get("overloads") is None:
        clean_data["overloads"] = []
    try:
        return ExtendedGhostRef.model_validate(clean_data)
    except Exception:
        return None


def _register_ghost_ref(
    sym: str,
    ref: ExtendedGhostRef,
    endpoints: set[str],
    item_map: dict[str, dict[str, object]],
    ref_map: dict[str, ExtendedGhostRef],
) -> None:
    """Register an ExtendedGhostRef and its aliases in engine lookup structures.

    Args:
        sym (str): Primary symbol name.
        ref (ExtendedGhostRef): ExtendedGhostRef instance.
        endpoints (set[str]): Destination endpoint set.
        item_map (dict[str, dict[str, object]]): Destination dictionary map.
        ref_map (dict[str, ExtendedGhostRef]): Destination model map.
    """
    endpoints.add(sym)
    ref_map[sym] = ref
    item_map[sym] = ref.model_dump()
    for key in (ref.api_path, ref.name):
        if key:
            endpoints.add(key)
            ref_map[key] = ref
            item_map[key] = ref.model_dump()
    if ref.aliases:
        for alias in ref.aliases:
            endpoints.add(alias)
            ref_map[alias] = ref
            item_map[alias] = ref.model_dump()


def _populate_from_grounding_engine(
    engine: object | None,
    target: BackendSnapshotTargetModel | None,
    endpoints: set[str],
    item_map: dict[str, dict[str, object]],
    ref_map: dict[str, ExtendedGhostRef],
) -> None:
    """Populate endpoints and ghost refs from delegating GroundingEngine if available.

    Args:
        engine (object | None): GroundingEngine instance.
        target (BackendSnapshotTargetModel | None): Backend snapshot target model.
        endpoints (set[str]): Destination set of API endpoint strings.
        item_map (dict[str, dict[str, object]]): Destination map from symbol to dump dict.
        ref_map (dict[str, ExtendedGhostRef]): Destination map from symbol to ghost ref.
    """
    if engine is None or target is None:
        return
    try:
        discover_fn = getattr(engine, "_discover_target_files", None)
        has_files: bool = bool(discover_fn(target.framework)) if discover_fn is not None else False
        if not has_files:
            return
        load_fn = getattr(engine, "load_target", None)
        if load_fn is None:
            return
        target_refs: dict[str, ExtendedGhostRef] = load_fn(target.framework)
        for sym, ref in target_refs.items():
            _register_ghost_ref(sym, ref, endpoints, item_map, ref_map)
    except Exception:
        pass


def _populate_from_snapshot_data(
    snapshot_data: dict[str, object] | list[object],
    endpoints: set[str],
    item_map: dict[str, dict[str, object]],
    ref_map: dict[str, ExtendedGhostRef],
) -> None:
    """Extract endpoints and hydrate ghost refs from parsed snapshot JSON data.

    Args:
        snapshot_data (dict[str, object] | list[object]): Parsed snapshot JSON structure.
        endpoints (set[str]): Destination set to populate with valid endpoint strings.
        item_map (dict[str, dict[str, object]]): Destination map from endpoint to raw item dict.
        ref_map (dict[str, ExtendedGhostRef]): Destination map from endpoint to hydrated ghost ref.
    """
    if isinstance(snapshot_data, dict):
        _extract_category_endpoints(snapshot_data.get("categories"), endpoints, item_map)
    elif isinstance(snapshot_data, list):
        for item in snapshot_data:
            if isinstance(item, dict):
                _extract_item_endpoints(item, endpoints, item_map)

    for sym, it in list(item_map.items()):
        if sym not in ref_map:
            hydrated: ExtendedGhostRef | None = _hydrate_ghost_ref(it)
            if hydrated is not None:
                ref_map[sym] = hydrated


def _param_to_dict(p: object) -> dict[str, object]:
    """Convert a parameter object or model into a dictionary specification.

    Args:
        p (object): Parameter model, dictionary, or object.

    Returns:
        dict[str, object]: Parameter spec dictionary.
    """
    dump_fn: object = getattr(p, "model_dump", None)
    if callable(dump_fn):
        dumped: object = dump_fn()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(p, dict):
        return p
    return {}


def _signatures_from_ghost_ref(ref: ExtendedGhostRef) -> list[list[dict[str, object]]]:
    """Extract parameter signatures and overloads from an ExtendedGhostRef.

    Args:
        ref (ExtendedGhostRef): Populated ExtendedGhostRef model.

    Returns:
        list[list[dict[str, object]]]: Extracted distinct signature parameter lists.
    """
    signatures: list[list[dict[str, object]]] = []
    if ref.params:
        primary_sig: list[dict[str, object]] = [_param_to_dict(p) for p in ref.params]
        signatures.extend(_split_signature_params(primary_sig))

    if ref.overloads:
        for ov in ref.overloads:
            ov_params: object = ov.params if hasattr(ov, "params") else (ov.get("params") if isinstance(ov, dict) else None)
            if isinstance(ov_params, list) and ov_params:
                ov_sig: list[dict[str, object]] = [_param_to_dict(p) for p in ov_params]
                signatures.extend(_split_signature_params(ov_sig))
    return signatures


def _signatures_from_item(item: dict[str, object]) -> list[list[dict[str, object]]]:
    """Extract fallback parameter signatures from a raw item dictionary.

    Args:
        item (dict[str, object]): Raw snapshot metadata item.

    Returns:
        list[list[dict[str, object]]]: Fallback signature parameter lists.
    """
    fallback_signatures: list[list[dict[str, object]]] = []
    params_obj: object = item.get("params")
    if isinstance(params_obj, list) and params_obj:
        fallback_signatures.extend(_split_signature_params(params_obj))

    overloads_obj: object = item.get("overloads")
    if isinstance(overloads_obj, list):
        for ov in overloads_obj:
            if isinstance(ov, dict) and isinstance(ov.get("params"), list) and ov["params"]:
                fallback_signatures.extend(_split_signature_params(ov["params"]))
    return fallback_signatures


class SnapshotGroundingEngine:
    """Grounding engine providing deterministic static verification against framework snapshots."""

    def __init__(
        self,
        snapshot_dir: str | None = None,
        config: BackendSnapshotTargetsConfig | None = None,
    ) -> None:
        """Initialize the SnapshotGroundingEngine.

        Args:
            snapshot_dir (str | None): Path to snapshot directory. Defaults to standard repository location.
            config (BackendSnapshotTargetsConfig | None): Optional loaded targets configuration.
        """
        if config is None:
            config = load_backend_snapshot_targets()
        self.config: BackendSnapshotTargetsConfig = config
        self._is_default_snapshot_dir: bool = snapshot_dir is None
        self.snapshot_dir: str = os.path.abspath(snapshot_dir) if snapshot_dir is not None else _resolve_default_snapshot_dir()

        self._endpoint_cache: dict[str, set[str]] = {}
        self._endpoint_item_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._ghost_ref_cache: dict[str, dict[str, ExtendedGhostRef]] = {}
        self._raw_snapshots: dict[str, dict[str, object] | list[object]] = {}
        self.frameworks_dir: str = os.path.abspath(os.path.join(os.path.dirname(self.snapshot_dir), "frameworks"))

        self._grounding_engine: object | None = None
        try:
            import importlib

            mod = importlib.import_module("ml_ecosystem_snapshots.grounding.engine")
            ge_cls = getattr(mod, "GroundingEngine", None)
            if ge_cls is not None:
                search_dirs: list[str] = [self.snapshot_dir, self.frameworks_dir]
                self._grounding_engine = ge_cls(base_dirs=search_dirs)
        except Exception:
            self._grounding_engine = None

    def _find_snapshot_in_dir(self, sdir: str, prefix: str, suffix: str) -> str | None:
        """Find the latest matching snapshot file in a directory (.json or .json.gz).

        Args:
            sdir (str): Path to candidate directory.
            prefix (str): File prefix to match.
            suffix (str): File suffix to match.

        Returns:
            str | None: Absolute path to the latest matching snapshot file, or None.
        """
        if not os.path.exists(sdir) or not os.path.isdir(sdir):
            return None
        all_candidates: list[str] = [f for f in os.listdir(sdir) if f.startswith(prefix) and (f.endswith(suffix) or f.endswith(f"{suffix}.gz"))]
        if not all_candidates:
            return None
        versioned: list[str] = sorted([f for f in all_candidates if "unknown" not in f])
        selected: str = versioned[-1] if versioned else sorted(all_candidates)[-1]
        return os.path.join(sdir, selected)

    def get_snapshot_path(self, backend_name: str) -> str | None:
        """Locate the latest static JSON snapshot file for a specified backend.

        Args:
            backend_name (str): Identifier of the target backend.

        Returns:
            str | None: Absolute file path to the snapshot JSON, or None if not found.
        """
        target: BackendSnapshotTargetModel | None = self.config.targets.get(backend_name)
        if target is None or target.status == "missing_upstream":
            return None

        prefix: str = target.snapshot_glob.split("*")[0]
        suffix: str = target.snapshot_glob.split("*")[-1]

        dirs_to_check: list[str] = [self.snapshot_dir]
        if self._is_default_snapshot_dir:
            candidate_dirs: list[str] = [
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "ml-ecosystem-snapshots",
                        "src",
                        "ml_ecosystem_snapshots",
                        "snapshots",
                    )
                ),
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "ml-ecosystem-snapshots",
                        "src",
                        "ml_framework_snapshots",
                        "snapshots",
                    )
                ),
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "ml-framework-snapshots",
                        "src",
                        "ml_framework_snapshots",
                        "snapshots",
                    )
                ),
                os.path.expanduser(os.path.join("~", ".cache", "ml_ecosystem_snapshots", "snapshots")),
                os.path.expanduser(os.path.join("~", ".cache", "ml_ecosystem_snapshots")),
                os.path.expanduser(os.path.join("~", ".cache", "ml_switcheroo_compiler", "snapshots")),
                os.path.expanduser(os.path.join("~", ".cache", "ml_framework_snapshots", "snapshots")),
            ]
            for cd in candidate_dirs:
                if cd not in dirs_to_check and os.path.exists(cd):
                    dirs_to_check.append(cd)

        for sdir in dirs_to_check:
            cand: str | None = self._find_snapshot_in_dir(sdir, prefix, suffix)
            if cand is not None and _is_snapshot_file_complete(cand, backend_name):
                return cand

        if target.framework_file and os.path.exists(self.frameworks_dir):
            candidate: str = os.path.join(self.frameworks_dir, target.framework_file)
            if os.path.exists(candidate):
                return candidate

        return None

    def load_snapshot(self, backend_name: str) -> dict[str, object] | list[object]:
        """Load and cache the raw JSON snapshot dictionary for a backend.

        Args:
            backend_name (str): Identifier of the target backend.

        Returns:
            dict[str, object] | list[object]: Parsed snapshot JSON structure.

        Raises:
            FileNotFoundError: If no snapshot file matches the configured glob.
        """
        if backend_name in self._raw_snapshots:
            return self._raw_snapshots[backend_name]

        path: str | None = self.get_snapshot_path(backend_name)
        if path is None:
            raise FileNotFoundError(f"No snapshot JSON file found for backend '{backend_name}' in {self.snapshot_dir}")

        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as gf:
                data: dict[str, object] | list[object] = json.load(gf)
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

        self._raw_snapshots[backend_name] = data
        return data

    def get_valid_endpoints(self, backend_name: str) -> set[str]:
        """Extract and cache all declared API endpoints from a backend snapshot.

        Args:
            backend_name (str): Identifier of the target backend.

        Returns:
            set[str]: Set of all canonical API paths, short names, and aliases in snapshot.
        """
        if backend_name in self._endpoint_cache:
            return self._endpoint_cache[backend_name]

        endpoints: set[str] = set()
        item_map: dict[str, dict[str, object]] = {}
        ref_map: dict[str, ExtendedGhostRef] = {}
        target: BackendSnapshotTargetModel | None = self.config.targets.get(backend_name)

        try:
            snapshot_data: dict[str, object] | list[object] = self.load_snapshot(backend_name)
        except FileNotFoundError:
            self._endpoint_cache[backend_name] = endpoints
            self._endpoint_item_cache[backend_name] = item_map
            self._ghost_ref_cache[backend_name] = ref_map
            return endpoints

        _populate_from_grounding_engine(self._grounding_engine, target, endpoints, item_map, ref_map)
        _populate_from_snapshot_data(snapshot_data, endpoints, item_map, ref_map)
        _extract_stub_endpoints(backend_name, endpoints)

        self._endpoint_cache[backend_name] = endpoints
        self._endpoint_item_cache[backend_name] = item_map
        self._ghost_ref_cache[backend_name] = ref_map
        return endpoints

    def get_endpoint_ghost_ref(self, backend_name: str, endpoint: str) -> ExtendedGhostRef | None:
        """Retrieve the canonical ExtendedGhostRef for an endpoint.

        Args:
            backend_name (str): Target backend identifier.
            endpoint (str): Fully qualified or short API path.

        Returns:
            ExtendedGhostRef | None: Populated ExtendedGhostRef model or None.
        """
        target: BackendSnapshotTargetModel | None = self.config.targets.get(backend_name)
        if target is None:
            return None

        self.get_valid_endpoints(backend_name)
        ref_map: dict[str, ExtendedGhostRef] = self._ghost_ref_cache.get(backend_name, {})

        candidates: list[str] = [endpoint]
        for root in target.canonical_roots:
            if endpoint.startswith(f"{root}."):
                suffix: str = endpoint[len(root) + 1 :]
                candidates.extend(
                    [
                        suffix,
                        *(f"{alt_root}.{suffix}" for alt_root in target.canonical_roots),
                        f"{target.framework}.{suffix}",
                        suffix.split(".")[-1],
                    ]
                )

        for cand in candidates:
            if cand in ref_map:
                return ref_map[cand]

        return None

    def get_endpoint_item(self, backend_name: str, endpoint: str) -> dict[str, object] | None:
        """Retrieve the raw snapshot metadata item dictionary for an endpoint.

        Args:
            backend_name (str): Target backend identifier.
            endpoint (str): Fully qualified or short API path.

        Returns:
            dict[str, object] | None: Snapshot entity dictionary or None if not found.
        """
        target: BackendSnapshotTargetModel | None = self.config.targets.get(backend_name)
        if target is None:
            return None

        self.get_valid_endpoints(backend_name)
        item_map: dict[str, dict[str, object]] = self._endpoint_item_cache.get(backend_name, {})

        candidates: list[str] = [endpoint]
        for root in target.canonical_roots:
            if endpoint.startswith(f"{root}."):
                suffix: str = endpoint[len(root) + 1 :]
                candidates.extend(
                    [
                        suffix,
                        *(f"{alt_root}.{suffix}" for alt_root in target.canonical_roots),
                        f"{target.framework}.{suffix}",
                        suffix.split(".")[-1],
                    ]
                )

        for cand in candidates:
            if cand in item_map:
                return item_map[cand]

        return None

    def get_endpoint_signatures(self, backend_name: str, endpoint: str) -> list[list[dict[str, object]]]:
        """Retrieve all declared parameter signatures (base and overloads) for an endpoint.

        Args:
            backend_name (str): Target backend identifier.
            endpoint (str): Fully qualified or short API path.

        Returns:
            list[list[dict[str, object]]]: List of parameter signatures.
        """
        ref: ExtendedGhostRef | None = self.get_endpoint_ghost_ref(backend_name, endpoint)
        if ref is not None:
            signatures: list[list[dict[str, object]]] = _signatures_from_ghost_ref(ref)
            if signatures:
                return signatures

        item: dict[str, object] | None = self.get_endpoint_item(backend_name, endpoint)
        if item is None:
            return []
        return _signatures_from_item(item)

    def get_endpoint_parameters(self, backend_name: str, endpoint: str) -> list[dict[str, object]] | None:
        """Retrieve list of declared parameter specifications for an endpoint.

        Args:
            backend_name (str): Target backend identifier.
            endpoint (str): Fully qualified or short API path.

        Returns:
            list[dict[str, object]] | None: List of parameter spec dictionaries, or None if endpoint not found.
        """
        item: dict[str, object] | None = self.get_endpoint_item(backend_name, endpoint)
        if item is None:
            return None
        signatures: list[list[dict[str, object]]] = self.get_endpoint_signatures(backend_name, endpoint)
        if not signatures:
            params_obj: object = item.get("params")
            if params_obj is None and item.get("overloads") is None:
                return None
            return []
        seen_names: set[str] = set()
        res: list[dict[str, object]] = []
        for sig in signatures:
            for p in sig:
                p_name: str = str(p.get("name") or "")
                if p_name and p_name not in seen_names:
                    seen_names.add(p_name)
                    res.append(p)
                elif not p_name:
                    res.append(p)
        return res

    def get_endpoint_return_type(self, backend_name: str, endpoint: str) -> str | None:
        """Retrieve declared return type annotation string for an endpoint.

        Args:
            backend_name (str): Target backend identifier.
            endpoint (str): Fully qualified or short API path.

        Returns:
            str | None: Return type annotation string, or None if not found or unannotated.
        """
        item: dict[str, object] | None = self.get_endpoint_item(backend_name, endpoint)
        if item is None:
            return None
        ret_type: object = item.get("returns_type")
        if isinstance(ret_type, str) and ret_type:
            return ret_type
        return None

    def is_endpoint_valid(self, backend_name: str, endpoint: str) -> bool:
        """Verify whether an API endpoint exists in the static framework snapshot.

        Args:
            backend_name (str): Target backend name.
            endpoint (str): Fully qualified or short API path (e.g. 'torch.add', 'jax.numpy.sin').

        Returns:
            bool: True if the endpoint is validated against the static snapshot, False otherwise.
        """
        target: BackendSnapshotTargetModel | None = self.config.targets.get(backend_name)
        if target is None:
            return False

        valid_set: set[str] = self.get_valid_endpoints(backend_name)
        if not valid_set:
            return False

        if endpoint in valid_set:
            return True

        for root in target.canonical_roots:
            if endpoint.startswith(f"{root}."):
                suffix: str = endpoint[len(root) + 1 :]
                candidates: tuple[str, ...] = (
                    suffix,
                    f"{target.framework}.{suffix}",
                    *(f"{alt_root}.{suffix}" for alt_root in target.canonical_roots),
                    suffix.split(".")[-1],
                )
                if any(cand in valid_set for cand in candidates):
                    return True

        return False

    def suggest_closest_endpoint(self, backend_name: str, candidate: str, max_distance: int = 4) -> str | None:
        """Suggest the closest valid endpoint for a typo candidate using Levenshtein distance.

        Args:
            backend_name (str): Identifier of the target backend.
            candidate (str): Mistyped endpoint candidate name.
            max_distance (int): Maximum allowable Levenshtein distance. Defaults to 4.

        Returns:
            str | None: Best matching valid endpoint, or None if no close match is found.
        """
        target: BackendSnapshotTargetModel | None = self.config.targets.get(backend_name)
        if target is not None and self._grounding_engine is not None:
            try:
                suggest_fn = getattr(self._grounding_engine, "suggest_closest_symbol", None)
                if suggest_fn is not None:
                    suggested: str | None = suggest_fn(target.framework, candidate, max_distance=max_distance)
                    if suggested:
                        return suggested
            except Exception:
                pass

        valid: set[str] = self.get_valid_endpoints(backend_name)
        if not valid:
            return None
        candidate_lower: str = candidate.lower()
        best_match: str | None = None
        min_dist: int = max_distance + 1
        for endpoint in sorted(valid):
            dist: int = compute_levenshtein(candidate_lower, endpoint.lower())
            if dist < min_dist:
                min_dist = dist
                best_match = endpoint
        return best_match

    def validate_parameter_contract(
        self,
        backend_name: str,
        endpoint: str,
        positional_args_count: int,
        keyword_arg_names: list[str],
        raise_on_error: bool = False,
    ) -> list[str]:
        """Validate parameter positions, kinds, defaults, and keywords against static snapshot schema.

        Args:
            backend_name (str): Target backend framework identifier.
            endpoint (str): Fully qualified or short API endpoint string.
            positional_args_count (int): Number of positional arguments provided in call.
            keyword_arg_names (list[str]): List of keyword argument names provided in call.
            raise_on_error (bool): If True, raises GroundingValidationError when violations occur.

        Returns:
            list[str]: List of validation error messages, or empty list if valid.

        Raises:
            GroundingValidationError: If raise_on_error is True and parameter validation fails.
        """
        signatures: list[list[dict[str, object]]] = self.get_endpoint_signatures(backend_name, endpoint)
        if not signatures:
            params: list[dict[str, object]] | None = self.get_endpoint_parameters(backend_name, endpoint)
            if not params:
                return []
            signatures = [params]

        all_sig_errors: list[list[str]] = []
        kw_set: set[str] = set(keyword_arg_names)
        all_params: list[dict[str, object]] = self.get_endpoint_parameters(backend_name, endpoint) or []
        endpoint_allowed_kwargs: set[str] = {str(p.get("name")) for p in all_params if p.get("kind") in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")}

        for sig in signatures:
            sig_errors: list[str] = []
            has_var_kw: bool = any(p.get("kind") == "VAR_KEYWORD" for p in sig)
            allowed_kwargs: set[str] = {str(p.get("name")) for p in sig if p.get("kind") in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")}
            pos_only_kwargs: set[str] = {str(p.get("name")) for p in sig if p.get("kind") == "POSITIONAL_ONLY"}

            sig_errors.extend(
                _validate_keyword_args(
                    keyword_arg_names,
                    allowed_kwargs,
                    pos_only_kwargs,
                    has_var_kw,
                    backend_name,
                    endpoint,
                    endpoint_allowed_kwargs=endpoint_allowed_kwargs,
                )
            )

            has_var_pos: bool = any(p.get("kind") == "VAR_POSITIONAL" for p in sig)
            pos_params: list[dict[str, object]] = [p for p in sig if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")]
            max_pos: int = len(pos_params)

            if not has_var_pos and positional_args_count > max_pos:
                sig_errors.append(f"Too many positional arguments ({positional_args_count} > {max_pos}) for endpoint '{endpoint}' in backend '{backend_name}'")

            min_required: int = 0
            for p in pos_params:
                p_name: str = str(p.get("name") or "")
                is_mandatory: bool = bool(p.get("is_mandatory", True))
                default_val: object = p.get("default")
                if is_mandatory and default_val is None and p_name not in kw_set:
                    min_required += 1

            if positional_args_count < min_required:
                sig_errors.append(f"Missing required arguments (expected at least {min_required}, got {positional_args_count}) for endpoint '{endpoint}' in backend '{backend_name}'")

            if not sig_errors:
                return []
            all_sig_errors.append(sig_errors)

        chosen_errors: list[str] = min(all_sig_errors, key=len) if all_sig_errors else []

        if chosen_errors and raise_on_error:
            raise GroundingValidationError("; ".join(chosen_errors))

        return chosen_errors
