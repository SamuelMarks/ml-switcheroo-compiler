"""Validation script asserting zero hallucinated parameters across all compiler backends."""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    SnapshotGroundingEngine,
)


def _normalize_code(code: str) -> str:
    """Normalize template placeholders in code string to valid Python identifiers.

    Args:
        code (str): Raw template or expression code string.

    Returns:
        str: Normalized Python code string.
    """
    clean: str = code.strip()
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", r"_var_\1", clean)


def _extract_calls_from_ast(expr_str: str) -> list[tuple[str, int, list[str]]]:
    """Extract function call endpoints, positional count, and keyword argument names from AST.

    Args:
        expr_str (str): Python expression or template string.

    Returns:
        list[tuple[str, int, list[str]]]: List of tuples (endpoint, pos_count, kwarg_names).
    """
    norm: str = _normalize_code(expr_str)
    calls: list[tuple[str, int, list[str]]] = []
    try:
        tree: ast.AST = ast.parse(norm)
    except SyntaxError:
        return calls

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
                    parts.append(curr.id)
                    parts.reverse()
                    ep = ".".join(parts)
                else:
                    ep = func.attr
            elif isinstance(func, ast.Name):
                ep = func.id

            if ep and not ep.startswith("_var_"):
                pos_count: int = len(node.args)
                kw_names: list[str] = [str(k.arg) for k in node.keywords if k.arg is not None]
                calls.append((ep, pos_count, kw_names))
    return calls


def validate_backend_mappings_parameters(
    engine: SnapshotGroundingEngine,
    backend_filter: str | None = None,
) -> list[str]:
    """Validate all backend mapping YAML templates against snapshot parameter contracts.

    Args:
        engine (SnapshotGroundingEngine): Framework snapshot verification engine.
        backend_filter (Optional[str]): Optional backend name to restrict validation.

    Returns:
        list[str]: Discovered parameter validation error messages.
    """
    errors: list[str] = []
    base_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "ml_switcheroo_compiler",
            "backends",
        )
    )
    mapping_files: list[str] = sorted(glob.glob(os.path.join(base_dir, "**", "mappings", "*.yaml"), recursive=True))

    for filepath in mapping_files:
        rel_path: str = os.path.relpath(filepath, base_dir)
        backend_name: str = rel_path.split(os.sep)[0]
        if backend_filter and backend_name != backend_filter:
            continue
        if backend_name not in engine.config.targets:
            continue
        if engine.config.targets[backend_name].status == "missing_upstream":
            continue

        with open(filepath, encoding="utf-8") as f:
            data: dict[str, object] | None = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue

        ops: dict[str, object] = data.get("operations", data)
        if not isinstance(ops, dict):
            continue

        for op_name, spec in ops.items():
            if not isinstance(spec, dict):
                continue
            ast_template: str = str(spec.get("ast_template") or "")
            custom_code: str = str(spec.get("custom_code") or "")

            for src in (ast_template, custom_code):
                if not src:
                    continue
                calls: list[tuple[str, int, list[str]]] = _extract_calls_from_ast(src)
                for ep, pos_count, kw_names in calls:
                    if not engine.is_endpoint_valid(backend_name, ep):
                        continue
                    errs: list[str] = engine.validate_parameter_contract(backend_name, ep, pos_count, kw_names)
                    for err in errs:
                        errors.append(f"{rel_path} ({op_name} -> {ep}): {err}")

    return errors


def validate_op_definitions_parameters(
    engine: SnapshotGroundingEngine,
    backend_filter: str | None = None,
) -> list[str]:
    """Validate all op definition variants against snapshot parameter contracts.

    Args:
        engine (SnapshotGroundingEngine): Framework snapshot verification engine.
        backend_filter (Optional[str]): Optional backend name to restrict validation.

    Returns:
        list[str]: Discovered parameter validation error messages.
    """
    errors: list[str] = []
    base_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "ml_switcheroo_compiler",
            "ops",
            "definitions",
        )
    )
    def_files: list[str] = sorted(glob.glob(os.path.join(base_dir, "*.yaml")))

    for filepath in def_files:
        filename: str = os.path.basename(filepath)
        with open(filepath, encoding="utf-8") as f:
            data: dict[str, object] | None = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue

        op_name: str = str(data.get("operation") or filename[:-5])
        variants: dict[str, object] = data.get("variants") or {}
        if not isinstance(variants, dict):
            continue

        for backend_name, var_spec in variants.items():
            if backend_filter and backend_name != backend_filter:
                continue
            if backend_name not in engine.config.targets:
                continue
            if engine.config.targets[backend_name].status == "missing_upstream":
                continue
            if not isinstance(var_spec, dict):
                continue

            generator_expr: str = str(var_spec.get("generator") or "")
            if not generator_expr:
                continue

            calls: list[tuple[str, int, list[str]]] = _extract_calls_from_ast(generator_expr)
            for ep, pos_count, kw_names in calls:
                if not engine.is_endpoint_valid(backend_name, ep):
                    continue
                errs: list[str] = engine.validate_parameter_contract(backend_name, ep, pos_count, kw_names)
                for err in errs:
                    errors.append(f"{filename} [{backend_name}] ({op_name} -> {ep}): {err}")

    return errors


def main() -> int:
    """Execute parameter validation across backend mappings and op definitions.

    Returns:
        int: Process return code (0 for pass, 1 for fail).
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Validate framework parameter contracts against static snapshot schemas.")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional backend framework identifier to filter validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict verification mode.",
    )
    args: argparse.Namespace = parser.parse_args()

    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    print("================================================================")
    print("      ZERO HALLUCINATED PARAMETERS VALIDATION SUITE             ")
    print("================================================================")
    print(f"Snapshot directory: {engine.snapshot_dir}")

    mapping_errors: list[str] = validate_backend_mappings_parameters(engine, backend_filter=args.backend)
    op_errors: list[str] = validate_op_definitions_parameters(engine, backend_filter=args.backend)

    all_errors: list[str] = mapping_errors + op_errors
    if all_errors:
        print(f"\nParameter validation FAILED: {len(all_errors)} issues detected:")
        for e in all_errors:
            print("  -", e)
        return 1

    print("\nParameter validation PASSED: Zero hallucinated parameters detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
