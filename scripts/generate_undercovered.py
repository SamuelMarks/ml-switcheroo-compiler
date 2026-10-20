"""Generate UNDERCOVERED.md containing a markdown checkbox list of undercovered files."""

from __future__ import annotations

import json
import os
import typing

CoverageSummary = dict[str, typing.Union[int, "float"]]
CoverageEntry = dict[str, typing.Union[CoverageSummary, "dict[str, dict[str, CoverageSummary]]"]]


def _is_file_undercovered(file_data: CoverageEntry) -> tuple[bool, float, int, int]:
    """Check if a single file's coverage summary or functions are undercovered.

    Args:
        file_data: Coverage dictionary for a single file.

    Returns:
        tuple[bool, float, int, int]: (is_undercovered, pct, missing_lines, missing_branches).
    """
    raw_summary = file_data.get("summary", {})
    summary: CoverageSummary = {k: v for k, v in raw_summary.items() if isinstance(v, (int, float))}
    pct = float(summary.get("percent_covered", 0.0))
    missing_lines = int(summary.get("missing_lines", 0))
    missing_branches = int(summary.get("missing_branches", 0))
    partial_branches = int(summary.get("num_partial_branches", 0))
    raw_funcs = file_data.get("functions", {})
    funcs_dict = raw_funcs
    funcs_undercovered = any(isinstance(fn_data, dict) and isinstance(fn_data.get("summary"), dict) and (int(fn_data["summary"].get("missing_lines", 0)) > 0 or int(fn_data["summary"].get("missing_branches", 0)) > 0) for fn_data in funcs_dict.values())
    is_under = pct < 100.0 or missing_lines > 0 or missing_branches > 0 or partial_branches > 0 or funcs_undercovered
    return is_under, pct, missing_lines, missing_branches


def generate_undercovered_list() -> list[tuple[str, float, int, int]]:
    """Scan coverage_extract.json and find all files with < 100% coverage.

    Returns:
        list[tuple[str, float, int, int]]: List of (path, percent, missing_lines, missing_branches).
    """
    json_path = os.path.abspath("coverage_extract.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Coverage file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    files_in_src: set[str] = {os.path.normpath(os.path.join(root, file)) for root, _, files in os.walk("src/ml_switcheroo_compiler") for file in files if file.endswith(".py")}

    undercovered: list[tuple[str, float, int, int]] = []
    covered_files: set[str] = set()

    for file_path, file_data in data.get("files", {}).items():
        norm_path = os.path.normpath(file_path)
        if not norm_path.startswith("src/ml_switcheroo_compiler"):
            continue
        covered_files.add(norm_path)
        is_under, pct, missing_lines, missing_branches = _is_file_undercovered(file_data)
        if is_under:
            undercovered.append((norm_path, pct, missing_lines, missing_branches))

    for missing_file in sorted(files_in_src - covered_files):
        undercovered.append((missing_file, 0.0, -1, -1))

    undercovered.sort(key=lambda x: x[0])
    return undercovered


def main() -> None:
    """Generate UNDERCOVERED.md file."""
    items = generate_undercovered_list()
    lines = [
        "# Files With Undercovered Test Paths (< 100% Coverage)",
        "",
        f"Total undercovered files: **{len(items)}**",
        "",
    ]

    for path, pct, missing_lines, missing_branches in items:
        if missing_lines == -1:
            details = "0.0% coverage - no test execution"
        else:
            details = f"{pct:.1f}% coverage, {missing_lines} missing lines, {missing_branches} missing branches"
        lines.append(f"- [ ] `{path}` ({details})")

    lines.append("")

    with open("UNDERCOVERED.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated UNDERCOVERED.md with {len(items)} files.")


if __name__ == "__main__":
    main()
