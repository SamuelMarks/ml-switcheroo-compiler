"""Validation script for Rules 4 and 5."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import yaml


def _record_op_backend(op_counts: dict[str, list[str]], op: str, be: str) -> None:
    """Record a backend supporting an operation if not already recorded.

    Args:
        op_counts (dict[str, list[str]]): Mapping of operation names to supporting backend names.
        op (str): Operation name.
        be (str): Backend identifier.
    """
    if op not in op_counts:
        op_counts[op] = []
    if be not in op_counts[op]:
        op_counts[op].append(be)


def validate_n_to_m(snapshot_dir: str | None = None) -> list[str]:
    """Validate Rule 4 (N-to-M utility rule).

    Args:
        snapshot_dir (str | None): Optional path to snapshot directory.

    Returns:
        list[str]: A list of error messages describing operations that violate Rule 4.
    """
    del snapshot_dir
    op_counts: dict[str, list[str]] = {}
    backend_maps: list[str] = glob.glob("src/ml_switcheroo_compiler/backends/**/mappings.yaml", recursive=True) + glob.glob("src/ml_switcheroo_compiler/backends/**/eager_mappings.yaml", recursive=True) + glob.glob("src/ml_switcheroo_compiler/backends/**/mappings/*.yaml", recursive=True)
    for bp in sorted(backend_maps):
        with open(bp, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue
        parts = os.path.normpath(bp).split(os.sep)
        fallback_be = parts[3] if len(parts) > 3 else ""
        if "operation" in data and isinstance(data["operation"], str):
            op = data["operation"]
            be = str(data.get("backend") or fallback_be)
            if data.get("target_api") or data.get("ast_template") or data.get("custom_code"):
                _record_op_backend(op_counts, op, be)
        else:
            be = str(data.get("backend_name") or fallback_be)
            ops_dict = data.get("operations", {})
            if isinstance(ops_dict, dict):
                for op, spec in ops_dict.items():
                    if isinstance(spec, dict) and (spec.get("target_api") or spec.get("ast_template") or spec.get("custom_code")):
                        _record_op_backend(op_counts, str(op), be)

    errors: list[str] = []
    for op, backends in op_counts.items():
        if len(backends) < 2:
            errors.append(f"Operation '{op}' violates Rule 4 (N-to-M). It is only supported by 1 backend ({backends[0]}).")

    return errors


def main() -> int:
    """Run rule validation.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Validate N-to-M universal utility rule.")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Path to snapshot directory.")
    args, _ = parser.parse_known_args()

    errors: list[str] = validate_n_to_m(snapshot_dir=args.snapshot_dir)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"Found {len(errors)} operations violating Rule 4.", file=sys.stderr)
        return 1

    print("Found 0 operations violating Rule 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
