"""Script to detect and gate against upstream ML framework snapshot drifts and breaking changes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import typing

# Ensure src is discoverable when invoked directly or via pre-commit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ml_switcheroo_ir import AttributeValue
from pydantic import BaseModel, Field

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    SnapshotGroundingEngine,
    _resolve_default_snapshot_dir,
)


class DiffResult(BaseModel):
    """Result of diffing two snapshots.

    Attributes:
        added (list[str]): List of added api_paths.
        removed (list[str]): List of removed api_paths.
        signature_changed (list[str]): List of api_paths with changed signatures.
        breaking_signature_changed (list[str]): List of api_paths with breaking signature changes.
        non_breaking_signature_changed (list[str]): List of api_paths with non-breaking signature changes.
    """

    added: list[str] = Field(default_factory=list, description="List of added api_paths.")
    removed: list[str] = Field(default_factory=list, description="List of removed api_paths.")
    signature_changed: list[str] = Field(default_factory=list, description="List of api_paths with changed signatures.")
    breaking_signature_changed: list[str] = Field(
        default_factory=list,
        description="List of api_paths with breaking signature changes.",
    )
    non_breaking_signature_changed: list[str] = Field(
        default_factory=list,
        description="List of api_paths with non-breaking signature changes.",
    )


def _is_breaking_param_change(
    p1_list: list[dict[str, AttributeValue]],
    p2_list: list[dict[str, AttributeValue]],
) -> bool:
    """Determine if parameter list modifications between two snapshots are breaking.

    Args:
        p1_list (list[dict[str, AttributeValue]]): Old parameter list.
        p2_list (list[dict[str, AttributeValue]]): New parameter list.

    Returns:
        bool: True if parameter changes are backwards-incompatible.
    """
    p1_map: dict[str, dict[str, AttributeValue]] = {str(p.get("name")): p for p in p1_list}
    p2_map: dict[str, dict[str, AttributeValue]] = {str(p.get("name")): p for p in p2_list}

    if any(name not in p2_map for name in p1_map):
        return True

    for name, p2 in p2_map.items():
        if name not in p1_map:
            if p2.get("default") is None and p2.get("kind") not in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                return True
        else:
            p1 = p1_map[name]
            if (p1.get("default") is not None and p2.get("default") is None) or p1.get("default") != p2.get("default"):
                return True
    return False


def _flatten_snapshot_items(snap: dict[str, AttributeValue]) -> dict[str, dict[str, AttributeValue]]:
    """Flatten a snapshot dictionary into a mapping of API paths to item dictionaries.

    Args:
        snap (dict[str, AttributeValue]): Parsed snapshot dictionary.

    Returns:
        dict[str, dict[str, AttributeValue]]: Flattened endpoint dictionary.
    """
    flattened: dict[str, dict[str, AttributeValue]] = {}
    categories_dict = snap.get("categories", {})
    if isinstance(categories_dict, dict):
        for items in categories_dict.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        key = str(item.get("api_path") or item.get("name") or "")
                        if key:
                            flattened[key] = item
    return flattened


def diff_snapshots(
    snap1: dict[str, AttributeValue],
    snap2: dict[str, AttributeValue],
) -> DiffResult:
    """Diff two snapshots to detect added, removed, and signature-changed endpoints.

    Args:
        snap1 (dict[str, AttributeValue]): Reference old snapshot dictionary.
        snap2 (dict[str, AttributeValue]): Candidate new snapshot dictionary.

    Returns:
        DiffResult: Structured diff result model.
    """
    flat1 = _flatten_snapshot_items(snap1)
    flat2 = _flatten_snapshot_items(snap2)

    added: list[str] = []
    removed: list[str] = [path for path in flat1 if path not in flat2]
    signature_changed: list[str] = []
    breaking_signature_changed: list[str] = []
    non_breaking_signature_changed: list[str] = []

    for path, item2 in flat2.items():
        if path not in flat1:
            added.append(path)
        else:
            item1 = flat1[path]
            raw_p1 = item1.get("params", [])
            raw_p2 = item2.get("params", [])
            p1 = [p for p in raw_p1 if isinstance(p, dict)] if isinstance(raw_p1, list) else []
            p2 = [p for p in raw_p2 if isinstance(p, dict)] if isinstance(raw_p2, list) else []
            sig1 = [(p.get("name"), p.get("kind"), p.get("default"), p.get("annotation")) for p in p1]
            sig2 = [(p.get("name"), p.get("kind"), p.get("default"), p.get("annotation")) for p in p2]
            if sig1 != sig2:
                signature_changed.append(path)
                if _is_breaking_param_change(p1, p2):
                    breaking_signature_changed.append(path)
                else:
                    non_breaking_signature_changed.append(path)

    return DiffResult(
        added=sorted(added),
        removed=sorted(removed),
        signature_changed=sorted(signature_changed),
        breaking_signature_changed=sorted(breaking_signature_changed),
        non_breaking_signature_changed=sorted(non_breaking_signature_changed),
    )


def generate_changelog(diff: DiffResult) -> str:
    """Generate a Markdown changelog from a DiffResult.

    Args:
        diff (DiffResult): Computed diff result model.

    Returns:
        str: Markdown formatted changelog string.
    """
    lines: list[str] = ["# Changelog Report", ""]
    if not diff.added and not diff.removed and not diff.signature_changed:
        lines.append("No changes detected.")
        return "\n".join(lines)

    for section_title, entries in (
        ("## Added", diff.added),
        ("## Removed (Breaking)", diff.removed),
        ("## Breaking Signature Changes", diff.breaking_signature_changed),
        ("## Non-Breaking Signature Changes", diff.non_breaking_signature_changed),
    ):
        if entries:
            lines.append(section_title)
            for path in entries:
                lines.append(f"- `{path}`")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def load_snapshot_file(file_path: str) -> dict[str, AttributeValue]:
    """Load and parse a JSON snapshot file from disk.

    Args:
        file_path (str): Path to snapshot JSON file.

    Returns:
        dict[str, AttributeValue]: Parsed JSON snapshot payload.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Snapshot file not found: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        data: dict[str, AttributeValue] = json.load(f)
    return data


def compare_snapshot_files(
    old_file: str,
    new_file: str,
) -> tuple[DiffResult, str]:
    """Compare two snapshot files and compute diff and markdown changelog.

    Args:
        old_file (str): Path to base reference snapshot file.
        new_file (str): Path to candidate newer snapshot file.

    Returns:
        tuple[DiffResult, str]: Tuple of DiffResult model and Markdown changelog string.
    """
    old_data: dict[str, AttributeValue] = load_snapshot_file(old_file)
    new_data: dict[str, AttributeValue] = load_snapshot_file(new_file)
    diff: DiffResult = diff_snapshots(old_data, new_data)
    changelog: str = generate_changelog(diff)
    return diff, changelog


def check_directory_drift(
    old_dir: str,
    new_dir: str,
) -> tuple[dict[str, DiffResult], list[str]]:
    """Compare matching snapshot files across two directories.

    Args:
        old_dir (str): Base snapshot directory.
        new_dir (str): Candidate newer snapshot directory.

    Returns:
        tuple[dict[str, DiffResult], list[str]]: Map of backend name to DiffResult and list of breaking issues.
    """
    results: dict[str, DiffResult] = {}
    breaking_issues: list[str] = []

    if not os.path.exists(old_dir) or not os.path.exists(new_dir):
        return results, breaking_issues

    if os.path.abspath(old_dir) == os.path.abspath(new_dir) and os.path.abspath(old_dir) == os.path.abspath(_resolve_default_snapshot_dir()):
        engine = SnapshotGroundingEngine(snapshot_dir=None)
        for backend_name, target in sorted(engine.config.targets.items()):
            has_fw_py = os.path.exists(os.path.join(engine.frameworks_dir, f"{target.framework}.py"))
            if target.status == "available" and engine.get_snapshot_path(backend_name) is None and not has_fw_py and not engine.get_valid_endpoints(backend_name):
                breaking_issues.append(f"{backend_name}: missing snapshot matching '{target.snapshot_glob}'")
        return results, breaking_issues

    old_files: set[str] = {f for f in os.listdir(old_dir) if f.endswith(".json")}
    new_files: set[str] = {f for f in os.listdir(new_dir) if f.endswith(".json")}

    common: set[str] = old_files.intersection(new_files)
    for fname in sorted(common):
        diff, _ = compare_snapshot_files(os.path.join(old_dir, fname), os.path.join(new_dir, fname))
        results[fname] = diff
        if diff.breaking_signature_changed:
            for item in diff.breaking_signature_changed:
                breaking_issues.append(f"{fname}: breaking signature change in '{item}'")
        if diff.removed:
            for item in diff.removed:
                breaking_issues.append(f"{fname}: removed endpoint '{item}'")

    return results, breaking_issues


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for snapshot drift checker.

    Args:
        args (list[str]): Command line argument strings.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Verify upstream ML framework snapshots for breaking API drift.",
    )
    parser.add_argument(
        "--old",
        default=None,
        help="Path to reference snapshot file or directory.",
    )
    parser.add_argument(
        "--new",
        default=None,
        help="Path to newer snapshot file or directory.",
    )
    parser.add_argument(
        "--allow-breaking",
        action="store_true",
        help="Allow breaking API changes without returning an error exit code.",
    )
    parser.add_argument(
        "--output-changelog",
        default=None,
        help="Optional destination path to write generated markdown changelog.",
    )
    return parser.parse_args(args)


def _resolve_target_path(raw_target: str | None, fallback: str) -> str:
    """Resolve a target snapshot path or directory.

    Args:
        raw_target (str | None): Explicit CLI target path.
        fallback (str): Fallback snapshot directory.

    Returns:
        str: Resolved target path.
    """
    target: str = raw_target if raw_target is not None else fallback
    if os.path.basename(os.path.normpath(target)) == "ml-ecosystem-snapshots" and (not os.path.isdir(target) or not any(f.endswith((".json", ".json.gz")) for f in os.listdir(target))):
        return str(_resolve_default_snapshot_dir())
    if os.path.isdir(target) and os.path.isdir(os.path.join(target, "snapshots")):
        return os.path.join(target, "snapshots")
    return target


def main(argv: list[str] | None = None, stdout: typing.TextIO | None = None) -> int:
    """CLI entry point for checking snapshot drift.

    Args:
        argv (list[str] | None): Optional explicit argument list. Defaults to sys.argv[1:].
        stdout (typing.TextIO | None): Optional output stream. Defaults to sys.stdout.

    Returns:
        int: Exit status code (0 for success, 1 on unallowed breaking changes).
    """
    out: typing.TextIO = stdout if stdout is not None else sys.stdout
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    old_target: str = _resolve_target_path(args.old, _resolve_default_snapshot_dir())
    new_target: str = _resolve_target_path(args.new, old_target)

    if not os.path.exists(old_target) or not os.path.exists(new_target):
        out.write(f"Error: Snapshot path not found: {old_target}\n")
        return 1

    if os.path.isfile(old_target) and os.path.isfile(new_target):
        diff, changelog = compare_snapshot_files(old_target, new_target)
        if args.output_changelog:
            with open(args.output_changelog, "w", encoding="utf-8") as f:
                f.write(changelog)
        out.write(changelog + "\n")
        if (diff.breaking_signature_changed or diff.removed) and not args.allow_breaking:
            out.write("Error: Breaking API changes detected in snapshot diff.\n")
            return 1
        return 0

    if os.path.isdir(old_target) and os.path.isdir(new_target):
        _, breaking = check_directory_drift(old_target, new_target)
        if breaking and not args.allow_breaking:
            out.write("Error: Breaking API changes detected across directories:\n")
            for b in breaking:
                out.write(f"  - {b}\n")
            return 1
        out.write("Directory snapshot drift check passed successfully.\n")
        return 0

    out.write("Error: Both --old and --new must be either files or directories.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
