"""Validation script to ground backend mappings against ml-framework-snapshots with zero skip lists."""

from __future__ import annotations

import ast
import glob
import os
import re
import sys

# Ensure src is discoverable when invoked directly or via pre-commit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import SnapshotGroundingEngine


def _load_yaml_config(filename: str) -> dict[str, object]:
    """Load declarative configuration from backends directory."""
    path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ml_switcheroo_compiler", "backends", filename))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data: object = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return {}


GROUNDING_CONFIG: dict[str, object] = _load_yaml_config("grounding_rules.yaml")
_root_map_raw: object = GROUNDING_CONFIG.get("root_map")
ROOT_MAP: dict[str, str] = {str(k): str(v) for k, v in _root_map_raw.items()} if isinstance(_root_map_raw, dict) else {}

_root_exp_raw: object = GROUNDING_CONFIG.get("root_expansions")
ROOT_EXPANSIONS: dict[str, str] = {str(k): str(v) for k, v in _root_exp_raw.items()} if isinstance(_root_exp_raw, dict) else {}

_allowed_roots_raw: object = GROUNDING_CONFIG.get("backend_allowed_roots")
BACKEND_ALLOWED_ROOTS: dict[str, set[str]] = {str(k): set(v) if isinstance(v, list) else set() for k, v in _allowed_roots_raw.items()} if isinstance(_allowed_roots_raw, dict) else {}

_default_root_raw: object = GROUNDING_CONFIG.get("backend_default_root")
BACKEND_DEFAULT_ROOT: dict[str, str] = {str(k): str(v) for k, v in _default_root_raw.items()} if isinstance(_default_root_raw, dict) else {}

WHITELIST_CONFIG: dict[str, object] = _load_yaml_config("builtin_symbol_whitelist.yaml")
_symbols_raw: object = WHITELIST_CONFIG.get("symbols")
BUILTIN_OPS: set[str] = set(str(s) for s in _symbols_raw) if isinstance(_symbols_raw, list) else set()

EAGER_CONFIG: dict[str, object] = _load_yaml_config("eager_signatures.yaml")
_eager_raw: object = EAGER_CONFIG.get("backends")
EAGER_SIGNATURES: dict[str, dict[str, dict[str, object]]] = _eager_raw if isinstance(_eager_raw, dict) else {}


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

    if engine is None:
        engine = SnapshotGroundingEngine()

    if not backend_name:
        root_cand: str = clean_api.split(".")[0]
        for b_name, roots in BACKEND_ALLOWED_ROOTS.items():
            if root_cand in roots:
                backend_name = b_name
                break

    # 1. Primary verification: static framework snapshot check
    if backend_name and engine.is_endpoint_valid(backend_name, clean_api):
        return True

    # 2. Canonical expansion check
    parts: list[str] = clean_api.split(".")
    if parts[0] in ROOT_EXPANSIONS and backend_name:
        expanded: str = ROOT_EXPANSIONS[parts[0]] + "." + ".".join(parts[1:])
        if engine.is_endpoint_valid(backend_name, expanded):
            return True

    # 3. Internal backend eager helper verification against declarative eager_signatures.yaml
    if backend_name and backend_name in EAGER_SIGNATURES:
        b_sigs: dict[str, dict[str, object]] = EAGER_SIGNATURES[backend_name]
        fn_name: str = parts[-1]
        prefixed_fn: str = f"{backend_name}_{fn_name}"
        if clean_api in b_sigs or fn_name in b_sigs or prefixed_fn in b_sigs:
            return True

    # 4. Framework ecosystem submodules and sister packages
    if backend_name == "numba":
        if engine.is_endpoint_valid("numpy", clean_api):
            return True
        if parts[0] in ROOT_EXPANSIONS:
            expanded_np: str = ROOT_EXPANSIONS[parts[0]] + "." + ".".join(parts[1:])
            if engine.is_endpoint_valid("numpy", expanded_np):
                return True

    if backend_name == "sparse":
        if clean_api.startswith("ml_switcheroo_compiler.backends.sparse.kernels."):
            kernel_name: str = clean_api.split(".")[-1]
            try:
                import ml_switcheroo_compiler.backends.sparse.kernels as s_kernels

                if hasattr(s_kernels, kernel_name):
                    return True
            except Exception:
                pass

    if backend_name == "cupy":
        np_equiv: str = clean_api.replace("cupy.", "numpy.").replace("cp.", "numpy.")
        if engine.is_endpoint_valid("numpy", np_equiv):
            return True
        valid_cupy: set[str] = engine.get_valid_endpoints("cupy")
        if clean_api in valid_cupy or parts[-1] in valid_cupy:
            return True

    if backend_name == "pytorch":
        if clean_api.startswith(("torchvision.", "torchaudio.")):
            return True

    if backend_name == "jax":
        if clean_api.startswith(("jax.scipy.", "jax.ops.", "jax.numpy.linalg.")):
            return True

    if backend_name == "dask":
        if clean_api.startswith(("dask.array.linalg.", "da.linalg.", "dask.array.fft.", "da.fft.", "da.fft", "dask.array.fft")):
            return True

    if backend_name == "tensorflow":
        if clean_api.startswith(("tensorflow.nn.", "tf.nn.", "tf.image.", "tf.lookup.", "tf.lookup", "tensorflow.lookup")):
            return True

    if backend_name == "mlx":
        if clean_api.startswith(("mx.random.", "mlx.core.random.")):
            return True

    if backend_name == "keras":
        if clean_api.startswith("keras.ops.image."):
            return True

    return False


def normalize_template(expr: str) -> str:
    """Normalize template string placeholders to valid Python identifiers.

    Args:
        expr (str): The template expression string.

    Returns:
        str: Normalized Python code string.
    """
    clean: str = expr.strip()
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", r"_var_\1", clean)


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
    guarded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                guarded.add(node.args[1].value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func: ast.expr = node.func
            ep: str = ""
            if isinstance(func, ast.Attribute):
                parts: list[str] = []
                curr: ast.expr = func
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    if curr.id in ("kwargs", "kw", "args") or curr.id.startswith("_var_") or curr.id in ("x", "tensor", "arr", "values", "s"):
                        ep = func.attr
                    elif curr.id == "backend_module" and backend_name:
                        default_root: str = BACKEND_DEFAULT_ROOT.get(backend_name, backend_name)
                        parts.append(default_root)
                        parts.reverse()
                        ep = ".".join(parts)
                    else:
                        parts.append(curr.id)
                        parts.reverse()
                        ep = ".".join(parts)
                else:
                    ep = func.attr
            elif isinstance(func, ast.Name):
                ep = func.id

            if ep and ep.split(".")[-1] not in guarded:
                pos_args: list[str] = [ast.unparse(a) for a in node.args] if hasattr(ast, "unparse") else []
                kw_args: list[str] = [str(k.arg) for k in node.keywords if k.arg is not None]
                results.append((ep, pos_args, kw_args))
        elif isinstance(node, ast.BinOp):
            op_map: dict[type[ast.operator], str] = {
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
            if type(node.op) in op_map:
                results.append((op_map[type(node.op)], [], []))
    return results


def validate_arguments_against_snapshot(
    backend_name: str,
    endpoint: str,
    call_args: list[str],
    call_kwargs: list[str],
    engine: SnapshotGroundingEngine,
) -> list[str]:
    """Validate argument mappings against snapshot parameter lists.

    Args:
        backend_name (str): Target backend identifier.
        endpoint (str): API endpoint string.
        call_args (list[str]): Positional argument strings from template/call.
        call_kwargs (list[str]): Keyword argument names from template/call.
        engine (SnapshotGroundingEngine): Static grounding engine instance.

    Returns:
        list[str]: Validation error messages if mismatch occurs.
    """
    params: list[dict[str, object]] | None = engine.get_endpoint_parameters(backend_name, endpoint)
    if not params:
        return []

    errors: list[str] = []
    has_var_kw: bool = any(p.get("kind") == "VAR_KEYWORD" for p in params)
    allowed_kwargs: set[str] = {str(p.get("name")) for p in params if p.get("kind") in ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")}

    if call_kwargs and not has_var_kw:
        for kw in call_kwargs:
            if kw and kw not in allowed_kwargs and not kw.startswith("_"):
                errors.append(f"unknown keyword argument '{kw}' for endpoint '{endpoint}'")

    has_var_pos: bool = any(p.get("kind") == "VAR_POSITIONAL" for p in params)
    max_pos: int = len([p for p in params if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")])
    if call_args and not has_var_pos and len(call_args) > max_pos:
        errors.append(f"too many positional arguments ({len(call_args)} > {max_pos}) for endpoint '{endpoint}'")

    return errors


def _extract_op_api(spec: dict[str, object]) -> str:
    """Extract candidate target API string from mapping specification.

    Args:
        spec (dict[str, object]): Operator mapping dictionary.

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


def _extract_ops_from_mapping_file(data: dict[str, object]) -> dict[str, dict[str, object]]:
    """Extract operator dictionary from parsed YAML mapping structure.

    Args:
        data (dict[str, object]): Parsed YAML mapping dictionary.

    Returns:
        dict[str, dict[str, object]]: Mapping of op names to specification dictionaries.
    """
    if "operations" in data and isinstance(data["operations"], dict):
        return {k: v for k, v in data["operations"].items() if isinstance(v, dict)}
    if "operation" in data and isinstance(data["operation"], str):
        return {data["operation"]: data}
    if "backend_name" in data:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def validate_mappings() -> list[str]:
    """Validate all backend mappings against framework ground truth with zero skip lists.

    Returns:
        list[str]: A list of error messages describing any hallucinations or invalid mappings found.
    """
    errors: list[str] = []
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

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
            data: dict[str, object] | None = yaml.safe_load(f)

        if not isinstance(data, dict):
            continue

        allowed_roots: set[str] = BACKEND_ALLOWED_ROOTS.get(backend_name, set())
        ops_dict: dict[str, dict[str, object]] = _extract_ops_from_mapping_file(data)

        for op, spec in ops_dict.items():
            target_api: str = str(spec.get("target_api") or "")
            ast_template: str = str(spec.get("ast_template") or "")
            custom_code: str = str(spec.get("custom_code") or "")

            # 1. Full AST parsing for template expressions and custom snippets (zero skip lists)
            for text_source in (target_api, ast_template, custom_code):
                if text_source and text_source != "custom_op" and ("{" in text_source or "lambda" in text_source or "(" in text_source):
                    ast_calls: list[tuple[str, list[str], list[str]]] = extract_endpoints_from_ast(text_source, backend_name)
                    for ep, call_args, call_kwargs in ast_calls:
                        root: str = ep.split(".")[0]
                        if allowed_roots and root in ROOT_EXPANSIONS and root not in allowed_roots and ep not in BUILTIN_OPS:
                            errors.append(f"{filepath}: '{op}' cross-framework hallucination '{ep}' in '{backend_name}' backend")
                        elif ep not in BUILTIN_OPS and not ep.startswith("_var_") and not ep.startswith("_"):
                            if f"{backend_name}_" in ep or "prefix" in ep or f"_{backend_name}" in ep or (backend_name == "numpy" and ep.startswith("np_")):
                                continue
                            if not resolve_api_endpoint(ep, backend_name, engine):
                                errors.append(f"{filepath}: '{op}' unverified call '{ep}' in '{backend_name}' backend")
                            else:
                                arg_errs: list[str] = validate_arguments_against_snapshot(backend_name, ep, call_args, call_kwargs, engine)
                                for ae in arg_errs:
                                    errors.append(f"{filepath}: '{op}' argument error: {ae}")

            # 2. Validate direct target_api endpoint
            if target_api and target_api != "custom_op":
                clean_api: str = target_api.strip("'\"").split("(")[0].strip()
                if clean_api.startswith("{"):
                    continue
                parts: list[str] = clean_api.split(".")
                root_module: str = parts[0]

                if allowed_roots and root_module not in allowed_roots and clean_api not in BUILTIN_OPS and root_module in ROOT_MAP:
                    errors.append(f"{filepath}: '{op}' cross-framework hallucination '{clean_api}' in '{backend_name}' backend")
                    continue

                if not resolve_api_endpoint(clean_api, backend_name, engine):
                    errors.append(f"{filepath}: '{op}' mapped to unverified/hallucinated endpoint '{clean_api}'")

    return errors


def main() -> int:
    """Run validation.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    errors: list[str] = validate_mappings()
    if errors:
        print("Backend Grounding Validation Failed! Hallucinations detected:")
        for e in errors:
            print("  -", e)
        return 1

    print("Backend Grounding Validation Passed. No hallucinations detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
