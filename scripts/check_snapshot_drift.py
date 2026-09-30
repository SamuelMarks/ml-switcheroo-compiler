"""Script to detect and gate against upstream ML framework snapshot drifts and breaking changes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TextIO

from ml_ecosystem_snapshots.diff import DiffResult, diff_snapshots, generate_changelog


def load_snapshot_file(file_path: str) -> dict[str, object]:
    """Load and parse a JSON snapshot file from disk.

    Args:
        file_path (str): Path to snapshot JSON file.

    Returns:
        dict[str, object]: Parsed JSON snapshot payload.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Snapshot file not found: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        data: dict[str, object] = json.load(f)
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
    old_data: dict[str, object] = load_snapshot_file(old_file)
    new_data: dict[str, object] = load_snapshot_file(new_file)
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


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    """Execute snapshot drift detection workflow.

    Args:
        argv (list[str] | None): Optional explicit argument list. Defaults to sys.argv[1:].
        stdout (TextIO | None): Optional output stream. Defaults to sys.stdout.

    Returns:
        int: Exit status code (0 for success, 1 on unallowed breaking changes).
    """
    out: TextIO = stdout if stdout is not None else sys.stdout
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    from ml_switcheroo_compiler.backends.snapshot_grounding import _resolve_default_snapshot_dir

    old_target: str = args.old if args.old is not None else _resolve_default_snapshot_dir()
    new_target: str = args.new if args.new is not None else old_target

    if os.path.isfile(old_target) and os.path.isfile(new_target):
        diff, changelog = compare_snapshot_files(old_target, new_target)
        if args.output_changelog:
            with open(args.output_changelog, "w", encoding="utf-8") as f:
                f.write(changelog)
        out.write(changelog + "\n")
        has_breaking: bool = bool(diff.breaking_signature_changed or diff.removed)
        if has_breaking and not args.allow_breaking:
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
