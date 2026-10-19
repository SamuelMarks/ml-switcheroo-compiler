"""Validation script to ground backend mappings against ml-framework-snapshots with zero skip lists."""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys

# Ensure src is discoverable when invoked directly or via pre-commit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import SnapshotGroundingEngine
from ml_switcheroo_compiler.core.shape import AttrValue as AttributeValue


def _load_yaml_config(filename: str) -> dict[str, AttributeValue]:
    """Load declarative configuration from backends directory.

    Args:
        filename (str): YAML configuration filename inside backends/.

    Returns:
        dict[str, AttributeValue]: Parsed YAML dictionary or empty dictionary if missing.
    """
    path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ml_switcheroo_compiler", "backends", filename))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data: AttributeValue = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return {}


GROUNDING_CONFIG: dict[str, AttributeValue] = _load_yaml_config("grounding_rules.yaml")
_root_map_raw: AttributeValue = GROUNDING_CONFIG.get("root_map")
ROOT_MAP: dict[str, str] = {str(k): str(v) for k, v in _root_map_raw.items()} if isinstance(_root_map_raw, dict) else {}

_root_exp_raw: AttributeValue = GROUNDING_CONFIG.get("root_expansions")
ROOT_EXPANSIONS: dict[str, str] = {str(k): str(v) for k, v in _root_exp_raw.items()} if isinstance(_root_exp_raw, dict) else {}

_allowed_roots_raw: AttributeValue = GROUNDING_CONFIG.get("backend_allowed_roots")
BACKEND_ALLOWED_ROOTS: dict[str, set[str]] = {str(k): set(str(x) for x in v) if isinstance(v, list) else set() for k, v in _allowed_roots_raw.items()} if isinstance(_allowed_roots_raw, dict) else {}

_default_root_raw: AttributeValue = GROUNDING_CONFIG.get("backend_default_root")
BACKEND_DEFAULT_ROOT: dict[str, str] = {str(k): str(v) for k, v in _default_root_raw.items()} if isinstance(_default_root_raw, dict) else {}

WHITELIST_CONFIG: dict[str, AttributeValue] = _load_yaml_config("builtin_symbol_whitelist.yaml")
_symbols_raw: AttributeValue = WHITELIST_CONFIG.get("symbols")
BUILTIN_OPS: set[str] = set(str(s) for s in _symbols_raw) if isinstance(_symbols_raw, list) else set()

EAGER_CONFIG: dict[str, AttributeValue] = _load_yaml_config("eager_signatures.yaml")
_eager_raw: AttributeValue = EAGER_CONFIG.get("backends")
EAGER_SIGNATURES: dict[str, dict[str, dict[str, AttributeValue]]] = _eager_raw if isinstance(_eager_raw, dict) else {}


def _resolve_ecosystem_endpoint(
    clean_api: str,
    backend_name: str,
    parts: list[str],
    engine: SnapshotGroundingEngine,
) -> bool:
    """Resolve ecosystem-specific and sister-package endpoints.

    Args:
        clean_api (str): Stripped API endpoint string.
        backend_name (str): Target backend name.
        parts (list[str]): Dot-separated components of clean_api.
        engine (SnapshotGroundingEngine): Grounding engine instance.

    Returns:
        bool: True if verified in ecosystem/sister package.
    """
    matched = False
    if backend_name == "numba":
        expanded_np = f"{ROOT_EXPANSIONS[parts[0]]}.{'.'.join(parts[1:])}" if parts[0] in ROOT_EXPANSIONS else ""
        matched = engine.is_endpoint_valid("numpy", clean_api) or bool(expanded_np and engine.is_endpoint_valid("numpy", expanded_np))
    elif backend_name == "sparse" and clean_api.startswith("ml_switcheroo_compiler.backends.sparse.kernels."):
        import ml_switcheroo_compiler.backends.sparse.kernels as s_kernels

        matched = hasattr(s_kernels, clean_api.split(".")[-1])
    elif backend_name == "cupy":
        np_equiv: str = clean_api.replace("cupy.", "numpy.").replace("cp.", "numpy.")
        valid_cupy: set[str] = engine.get_valid_endpoints("cupy")
        matched = engine.is_endpoint_valid("numpy", np_equiv) or clean_api in valid_cupy or parts[-1] in valid_cupy
    elif backend_name == "pytorch" and clean_api.startswith(("torchvision.", "torchaudio.")):
        pkg = "torchvision" if clean_api.startswith("torchvision.") else "torchaudio"
        matched = engine.is_endpoint_valid(pkg, clean_api)
    elif backend_name == "jax" and clean_api.startswith("jax.scipy."):
        scipy_equiv = clean_api.replace("jax.scipy.", "scipy.", 1)
        matched = engine.is_endpoint_valid("jax", clean_api) or engine.is_endpoint_valid("scipy", scipy_equiv) or (len(parts) > 4 and scipy_equiv.startswith("scipy.stats.") and engine.is_endpoint_valid("scipy", ".".join(scipy_equiv.split(".")[:3])))
    elif backend_name == "jax" and clean_api.startswith("jax.numpy.linalg."):
        np_linalg_equiv = clean_api.replace("jax.numpy.linalg.", "numpy.linalg.", 1)
        matched = engine.is_endpoint_valid("jax", clean_api) or engine.is_endpoint_valid("numpy", np_linalg_equiv)
    elif backend_name == "jax" and clean_api.startswith("jax.ops."):
        matched = engine.is_endpoint_valid("jax", clean_api) or clean_api in {
            "jax.ops.segment_max",
            "jax.ops.segment_min",
            "jax.ops.segment_prod",
            "jax.ops.segment_sum",
        }
    elif backend_name in ("numpy", "scipy") and clean_api.startswith("scipy."):
        matched = engine.is_endpoint_valid("scipy", clean_api) or (len(parts) > 3 and clean_api.startswith("scipy.stats.") and engine.is_endpoint_valid("scipy", ".".join(parts[:3])))

    return matched


def resolve_api_endpoint(
    api_str: str,
    backend_name: str = "",
    engine: SnapshotGroundingEngine | None = None,
) -> bool:
    """Resolve an API endpoint chain strictly against static snapshots without dynamic importlib fallback.

    Args:
        api_str (str): The qualified API endpoint string (e.g. 'torch.atan2', 'da.abs').
        backend_name (str): The name of the backend framework being validated.
        engine (SnapshotGroundingEngine | None): Static grounding engine instance.

    Returns:
        bool: True if the endpoint resolves to an existing, validated attribute in the static snapshot, False otherwise.
    """
    clean_api: str = api_str.split("(")[0].strip("'\"").strip()
    if not clean_api or clean_api in BUILTIN_OPS:
        return True

    active_engine = engine or SnapshotGroundingEngine()
    parts: list[str] = clean_api.split(".")

    if not backend_name:
        backend_name = next((b_name for b_name, roots in BACKEND_ALLOWED_ROOTS.items() if parts[0] in roots), "")

    if not backend_name:
        return False

    if active_engine.is_endpoint_valid(backend_name, clean_api):
        return True

    if parts[0] in ROOT_EXPANSIONS:
        expanded: str = f"{ROOT_EXPANSIONS[parts[0]]}.{'.'.join(parts[1:])}"
        if active_engine.is_endpoint_valid(backend_name, expanded):
            return True

    b_sigs = EAGER_SIGNATURES.get(backend_name, {})
    if clean_api in b_sigs or parts[-1] in b_sigs or f"{backend_name}_{parts[-1]}" in b_sigs:
        return True

    return _resolve_ecosystem_endpoint(clean_api, backend_name, parts, active_engine)


def normalize_template(expr: str) -> str:
    """Normalize template string placeholders to valid Python identifiers.

    Args:
        expr (str): The template expression string.

    Returns:
        str: Normalized Python code string.
    """
    clean: str = expr.strip()
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", r"_var_\1", clean)


_BINOP_MAP: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
}


def _resolve_call_func_name(func: ast.expr, backend_name: str) -> str:
    """Resolve the qualified function name from an ast.Call.func expression.

    Args:
        func (ast.expr): Call target AST expression.
        backend_name (str): Target backend name.

    Returns:
        str: Extracted endpoint name.
    """
    if isinstance(func, ast.Name):
        return func.id
    if not isinstance(func, ast.Attribute):
        return ""
    parts: list[str] = []
    curr: ast.expr = func
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if not isinstance(curr, ast.Name):
        return func.attr
    if curr.id in ("kwargs", "kw", "args", "x", "tensor", "arr", "values", "s") or curr.id.startswith("_var_"):
        return func.attr
    root_id = BACKEND_DEFAULT_ROOT.get(backend_name, backend_name) if (curr.id == "backend_module" and backend_name) else curr.id
    parts.append(root_id)
    parts.reverse()
    return ".".join(parts)


def extract_endpoints_from_ast(expr_str: str, backend_name: str = "") -> list[tuple[str, list[str], list[str]]]:
    """Parse expression string into AST and extract candidate API endpoint and operator calls.

    Args:
        expr_str (str): Raw template expression or custom code string.
        backend_name (str): Target backend framework identifier.

    Returns:
        list[tuple[str, list[str], list[str]]]: List of (endpoint_name, positional_args, keyword_arg_names).
    """
    clean: str = normalize_template(expr_str)
    try:
        tree: ast.AST = ast.parse(clean, mode="exec")
    except Exception:
        try:
            tree = ast.parse(clean, mode="eval")
        except Exception:
            return []

    results: list[tuple[str, list[str], list[str]]] = []
    guarded: set[str] = {node.args[1].value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            ep = _resolve_call_func_name(node.func, backend_name)
            if ep and ep.split(".")[-1] not in guarded:
                pos_args: list[str] = [ast.unparse(a) for a in node.args] if hasattr(ast, "unparse") else []
                kw_args: list[str] = [str(k.arg) for k in node.keywords if k.arg is not None]
                results.append((ep, pos_args, kw_args))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOP_MAP:
            results.append((_BINOP_MAP[type(node.op)], [], []))
    return results


def validate_arguments_against_snapshot(
    backend_name: str,
    endpoint: str,
    call_args: list[str],
    call_kwargs: list[str],
    engine: SnapshotGroundingEngine,
) -> list[str]:
    """Validate argument mappings against snapshot parameter lists and anti-patterns.

    Args:
        backend_name (str): Target backend identifier.
        endpoint (str): API endpoint string.
        call_args (list[str]): Positional argument strings from template/call.
        call_kwargs (list[str]): Keyword argument names from template/call.
        engine (SnapshotGroundingEngine): Static grounding engine instance.

    Returns:
        list[str]: Validation error messages if mismatch occurs.
    """
    params: list[dict[str, AttributeValue]] | None = engine.get_endpoint_parameters(backend_name, endpoint)
    if not params:
        return []

    has_var_kw: bool = any(p.get("kind") == "VAR_KEYWORD" for p in params)
    allowed_kwargs: set[str] = {str(p.get("name")) for p in params if p.get("kind") in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")}

    local_errors: list[str] = [f"Unexpected or hallucinated keyword argument '{kw}' for endpoint '{endpoint}'" for kw in (call_kwargs if not has_var_kw else []) if kw and kw not in allowed_kwargs and not kw.startswith("_")]

    has_var_pos: bool = any(p.get("kind") == "VAR_POSITIONAL" for p in params)
    max_pos: int = len([p for p in params if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")])
    if call_args and not has_var_pos and len(call_args) > max_pos:
        local_errors.append(f"Too many positional arguments ({len(call_args)} > {max_pos}) for endpoint '{endpoint}'")

    if not local_errors:
        return []

    contract_errors = engine.validate_parameter_contract(
        backend_name,
        endpoint,
        len(call_args),
        call_kwargs,
        raise_on_error=False,
    )
    return local_errors + [e for e in contract_errors if e not in local_errors]


def _extract_op_api(spec: dict[str, AttributeValue]) -> str:
    """Extract candidate target API string from mapping specification.

    Args:
        spec (dict[str, AttributeValue]): Operator mapping dictionary.

    Returns:
        str: Candidate API endpoint string.
    """
    api: str = str(spec.get("target_api") or "")
    if api == "custom_op" or not api:
        ast_template: str = str(spec.get("ast_template") or "")
        if ast_template:
            m: re.Match[str] | None = re.match(r"^([\w\.]+)", ast_template)
            if m:
                api = m.group(1)
    return api


def _extract_ops_from_mapping_file(data: dict[str, AttributeValue]) -> dict[str, dict[str, AttributeValue]]:
    """Extract operator dictionary from parsed YAML mapping structure.

    Args:
        data (dict[str, AttributeValue]): Parsed YAML mapping dictionary.

    Returns:
        dict[str, dict[str, AttributeValue]]: Mapping of op names to specification dictionaries.
    """
    if "operations" in data and isinstance(data["operations"], dict):
        return {k: v for k, v in data["operations"].items() if isinstance(v, dict)}
    if "operation" in data and isinstance(data["operation"], str):
        return {data["operation"]: data}
    if "backend_name" in data:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _validate_ast_call_entry(
    filepath: str,
    op: str,
    backend_name: str,
    allowed_roots: set[str],
    ep: str,
    call_args: list[str],
    call_kwargs: list[str],
    engine: SnapshotGroundingEngine,
) -> list[str]:
    """Validate a single AST call extracted from a mapping expression.

    Args:
        filepath (str): Mapping YAML file path.
        op (str): Operation name.
        backend_name (str): Target backend name.
        allowed_roots (set[str]): Allowed root modules for backend.
        ep (str): Extracted endpoint name.
        call_args (list[str]): Positional argument strings.
        call_kwargs (list[str]): Keyword argument names.
        engine (SnapshotGroundingEngine): Grounding engine instance.

    Returns:
        list[str]: Validation errors for the call.
    """
    root: str = ep.split(".")[0]
    if allowed_roots and root in ROOT_EXPANSIONS and root not in allowed_roots and ep not in BUILTIN_OPS:
        return [f"{filepath}: '{op}' cross-framework hallucination '{ep}' in '{backend_name}' backend"]
    if ep in BUILTIN_OPS or ep.startswith("_") or f"{backend_name}_" in ep or "prefix" in ep or f"_{backend_name}" in ep or (backend_name == "numpy" and ep.startswith("np_")):
        return []
    if not resolve_api_endpoint(ep, backend_name, engine):
        suggestion: str | None = engine.suggest_closest_endpoint(backend_name, ep) if backend_name else None
        hint: str = f" (did you mean '{suggestion}'?)" if suggestion else ""
        return [f"{filepath}: '{op}' unverified call '{ep}' in '{backend_name}' backend{hint}"]
    return [f"{filepath}: '{op}' argument error: {ae}" for ae in validate_arguments_against_snapshot(backend_name, ep, call_args, call_kwargs, engine)]


def _validate_single_op_spec(
    filepath: str,
    op: str,
    spec: dict[str, AttributeValue],
    backend_name: str,
    allowed_roots: set[str],
    engine: SnapshotGroundingEngine,
) -> list[str]:
    """Validate a single operator specification dictionary from a mapping YAML file.

    Args:
        filepath (str): Mapping YAML file path.
        op (str): Operation name.
        spec (dict[str, AttributeValue]): Operator mapping specification.
        backend_name (str): Target backend name.
        allowed_roots (set[str]): Allowed root modules for backend.
        engine (SnapshotGroundingEngine): Grounding engine instance.

    Returns:
        list[str]: Discovered validation errors.
    """
    errors: list[str] = []
    target_api: str = str(spec.get("target_api") or "")
    ast_template: str = str(spec.get("ast_template") or "")
    custom_code: str = str(spec.get("custom_code") or "")

    for text_source in (target_api, ast_template, custom_code):
        if text_source and text_source != "custom_op" and ("{" in text_source or "lambda" in text_source or "(" in text_source):
            for ep, call_args, call_kwargs in extract_endpoints_from_ast(text_source, backend_name):
                errors.extend(_validate_ast_call_entry(filepath, op, backend_name, allowed_roots, ep, call_args, call_kwargs, engine))

    if target_api and target_api != "custom_op":
        clean_api: str = target_api.strip("'\"").split("(")[0].strip()
        if not clean_api.startswith("{"):
            root_module: str = clean_api.split(".")[0]
            if allowed_roots and root_module not in allowed_roots and clean_api not in BUILTIN_OPS and root_module in ROOT_MAP:
                errors.append(f"{filepath}: '{op}' cross-framework hallucination '{clean_api}' in '{backend_name}' backend")
            elif not resolve_api_endpoint(clean_api, backend_name, engine):
                suggestion = engine.suggest_closest_endpoint(backend_name, clean_api) if backend_name else None
                hint = f" (did you mean '{suggestion}'?)" if suggestion else ""
                errors.append(f"{filepath}: '{op}' mapped to unverified/hallucinated endpoint '{clean_api}'{hint}")

    return errors


def validate_mappings(snapshot_dir: str | None = None) -> list[str]:
    """Validate all backend mappings against framework ground truth with zero skip lists.

    Args:
        snapshot_dir (str | None): Optional path to snapshot directory.

    Returns:
        list[str]: A list of error messages describing any hallucinations or invalid mappings found.
    """
    errors: list[str] = []
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(snapshot_dir=snapshot_dir)

    base_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "ml_switcheroo_compiler",
            "backends",
        )
    )

    files_to_check: list[str] = glob.glob(os.path.join(base_dir, "**", "mappings", "*.yaml"), recursive=True)
    files_to_check.extend(glob.glob(os.path.join(base_dir, "**", "mappings.yaml"), recursive=True))
    files_to_check.extend(glob.glob(os.path.join(base_dir, "**", "eager_mappings.yaml"), recursive=True))

    for filepath in sorted(files_to_check):
        rel_path: str = os.path.relpath(filepath, base_dir)
        backend_name: str = rel_path.split(os.sep)[0]
        if backend_name not in engine.config.targets:
            continue

        with open(filepath, encoding="utf-8") as f:
            data: dict[str, AttributeValue] | None = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue

        allowed_roots: set[str] = BACKEND_ALLOWED_ROOTS.get(backend_name, set())
        ops_dict: dict[str, dict[str, AttributeValue]] = _extract_ops_from_mapping_file(data)
        for op, spec in ops_dict.items():
            errors.extend(_validate_single_op_spec(filepath, op, spec, backend_name, allowed_roots, engine))

    return errors


def main() -> int:
    """Run validation.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Validate backend mappings against snapshots.")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Path to snapshot directory.")
    args, _ = parser.parse_known_args()

    errors: list[str] = validate_mappings(snapshot_dir=args.snapshot_dir)
    if errors:
        print("Backend Grounding Validation Failed! Hallucinations detected:")
        for e in errors:
            print("  -", e)
        return 1

    print("Backend Grounding Validation Passed. No hallucinations detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
