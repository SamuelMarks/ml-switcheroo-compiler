"""Static Grounding Engine for Backend API Mappings using ml-framework-snapshots.

This module provides schema-validated utilities to verify backend mapping declarations
against static JSON snapshots extracted by the ml-framework-snapshots library.
"""

from __future__ import annotations

import json
import os

import yaml
from pydantic import BaseModel, Field


class BackendSnapshotTargetModel(BaseModel):
    """Pydantic specification for a target backend framework snapshot binding."""

    framework: str = Field(description="Canonical framework name in ml-framework-snapshots.")
    snapshot_glob: str = Field(description="Glob pattern for locating JSON snapshot file.")
    status: str = Field(
        default="available",
        description="Snapshot status in ml-framework-snapshots ('available' or 'missing_upstream').",
    )
    framework_file: str | None = Field(
        default=None,
        description="Optional framework specification file in ml_framework_snapshots/frameworks.",
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


DISCREPANCY_PAIRS: list[tuple[str, str]] = [
    ("dim", "axis"),
    ("axis", "dim"),
    ("keepdim", "keepdims"),
    ("keepdims", "keepdim"),
    ("split_size_or_sections", "indices_or_sections"),
]


def _validate_keyword_args(
    keyword_arg_names: list[str],
    allowed_kwargs: set[str],
    has_var_kw: bool,
    backend_name: str,
    endpoint: str,
) -> list[str]:
    """Validate keyword arguments against allowed kwargs and check discrepancy pairs.

    Args:
        keyword_arg_names (list[str]): List of keyword argument names.
        allowed_kwargs (set[str]): Allowed keyword arguments.
        has_var_kw (bool): Whether backend endpoint accepts arbitrary var keywords.
        backend_name (str): Target backend framework identifier.
        endpoint (str): Fully qualified endpoint string.

    Returns:
        list[str]: Validation error messages.
    """
    errors: list[str] = []
    if has_var_kw:
        return errors

    for kw in keyword_arg_names:
        if not kw or kw in allowed_kwargs or kw.startswith("_"):
            continue
        for provided, expected in DISCREPANCY_PAIRS:
            if kw == provided and expected in allowed_kwargs:
                errors.append(f"Endpoint '{endpoint}' in backend '{backend_name}' does not accept keyword '{kw}'. Did you mean '{expected}'?")
                break
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

        if snapshot_dir is None:
            env_dir: str | None = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR")
            if env_dir and os.path.exists(env_dir):
                self.snapshot_dir = os.path.abspath(env_dir)
            else:
                compiler_cache_dir: str = os.path.expanduser(os.path.join("~", ".cache", "ml_switcheroo_compiler", "snapshots"))
                default_dir: str = os.path.abspath(
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
                )
                cache_dir: str = os.path.expanduser(os.path.join("~", ".cache", "ml_framework_snapshots", "snapshots"))
                if os.path.exists(compiler_cache_dir) and any(f.endswith(".json") for f in os.listdir(compiler_cache_dir)):
                    self.snapshot_dir = compiler_cache_dir
                else:
                    default_has_jsons: bool = os.path.exists(default_dir) and any(f.endswith(".json") for f in os.listdir(default_dir))
                    if not default_has_jsons and os.path.exists(cache_dir) and any(f.endswith(".json") for f in os.listdir(cache_dir)):
                        self.snapshot_dir = cache_dir
                    else:
                        self.snapshot_dir = default_dir
        else:
            self.snapshot_dir = os.path.abspath(snapshot_dir)

        self._endpoint_cache: dict[str, set[str]] = {}
        self._endpoint_item_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._raw_snapshots: dict[str, dict[str, object] | list[object]] = {}
        self.frameworks_dir: str = os.path.abspath(os.path.join(os.path.dirname(self.snapshot_dir), "frameworks"))

    def _find_snapshot_in_dir(self, sdir: str, prefix: str, suffix: str) -> str | None:
        """Find the latest matching snapshot file in a directory.

        Args:
            sdir (str): Path to candidate directory.
            prefix (str): File prefix to match.
            suffix (str): File suffix to match.

        Returns:
            str | None: Absolute path to the latest matching snapshot file, or None.
        """
        if not os.path.exists(sdir):
            return None
        all_candidates: list[str] = [f for f in os.listdir(sdir) if f.startswith(prefix) and f.endswith(suffix)]
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
            for extra in ("ml_switcheroo_compiler", "ml_framework_snapshots"):
                cd: str = os.path.expanduser(os.path.join("~", ".cache", extra, "snapshots"))
                if cd not in dirs_to_check and os.path.exists(cd):
                    dirs_to_check.append(cd)

        for sdir in dirs_to_check:
            cand: str | None = self._find_snapshot_in_dir(sdir, prefix, suffix)
            if cand is not None:
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

        with open(path, encoding="utf-8") as f:
            data: dict[str, object] | list[object] = json.load(f)

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
        try:
            snapshot_data: dict[str, object] | list[object] = self.load_snapshot(backend_name)
        except FileNotFoundError:
            self._endpoint_cache[backend_name] = endpoints
            self._endpoint_item_cache[backend_name] = item_map
            return endpoints

        if isinstance(snapshot_data, dict):
            _extract_category_endpoints(snapshot_data.get("categories"), endpoints, item_map)
        elif isinstance(snapshot_data, list):
            for item in snapshot_data:
                if isinstance(item, dict):
                    _extract_item_endpoints(item, endpoints, item_map)
        _extract_stub_endpoints(backend_name, endpoints)
        self._endpoint_cache[backend_name] = endpoints
        self._endpoint_item_cache[backend_name] = item_map
        return endpoints

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

        if endpoint in item_map:
            return item_map[endpoint]

        for root in target.canonical_roots:
            if endpoint.startswith(f"{root}."):
                suffix: str = endpoint[len(root) + 1 :]
                if suffix in item_map:
                    return item_map[suffix]
                for alt_root in target.canonical_roots:
                    cand: str = f"{alt_root}.{suffix}"
                    if cand in item_map:
                        return item_map[cand]
                canonical: str = f"{target.framework}.{suffix}"
                if canonical in item_map:
                    return item_map[canonical]

        return None

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
        params_obj: object = item.get("params")
        res: list[dict[str, object]] = []
        if isinstance(params_obj, list):
            res.extend([p for p in params_obj if isinstance(p, dict)])
        overloads_obj: object = item.get("overloads")
        if isinstance(overloads_obj, list):
            for ov in overloads_obj:
                if isinstance(ov, dict) and isinstance(ov.get("params"), list):
                    res.extend([p for p in ov["params"] if isinstance(p, dict)])
        if not res and params_obj is None:
            return None
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
                )
                if any(cand in valid_set for cand in candidates):
                    return True

        return False

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
        errors: list[str] = []
        params: list[dict[str, object]] | None = self.get_endpoint_parameters(backend_name, endpoint)
        if not params:
            return errors

        has_var_kw: bool = any(p.get("kind") == "VAR_KEYWORD" for p in params)
        allowed_kwargs: set[str] = {str(p.get("name")) for p in params if p.get("kind") in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")}

        errors.extend(
            _validate_keyword_args(
                keyword_arg_names,
                allowed_kwargs,
                has_var_kw,
                backend_name,
                endpoint,
            )
        )

        has_var_pos: bool = any(p.get("kind") == "VAR_POSITIONAL" for p in params)
        pos_params: list[dict[str, object]] = [p for p in params if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")]
        max_pos: int = len(pos_params)

        if not has_var_pos and positional_args_count > max_pos:
            errors.append(f"Too many positional arguments ({positional_args_count} > {max_pos}) for endpoint '{endpoint}' in backend '{backend_name}'")

        # Calculate minimum required positional arguments not supplied via kwargs
        kw_set: set[str] = set(keyword_arg_names)
        min_required: int = 0
        for p in pos_params:
            p_name: str = str(p.get("name") or "")
            is_mandatory: bool = bool(p.get("is_mandatory", True))
            default_val: object = p.get("default")
            if is_mandatory and default_val is None and p_name not in kw_set:
                min_required += 1

        if positional_args_count < min_required:
            errors.append(f"Missing required arguments (expected at least {min_required}, got {positional_args_count}) for endpoint '{endpoint}' in backend '{backend_name}'")

        if errors and raise_on_error:
            raise GroundingValidationError("; ".join(errors))

        return errors
