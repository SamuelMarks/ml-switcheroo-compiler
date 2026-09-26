"""Generate UNDERCOVERED.md containing a markdown checkbox list of undercovered files."""

from __future__ import annotations

import json
import os


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

    files_in_src: set[str] = set()
    for root, _, files in os.walk("src/ml_switcheroo_compiler"):
        for file in files:
            if file.endswith(".py"):
                files_in_src.add(os.path.normpath(os.path.join(root, file)))

    undercovered: list[tuple[str, float, int, int]] = []
    covered_files: set[str] = set()

    for file_path, file_data in data.get("files", {}).items():
        norm_path = os.path.normpath(file_path)
        if not norm_path.startswith("src/ml_switcheroo_compiler"):
            continue
        covered_files.add(norm_path)
        summary = file_data.get("summary", {})
        pct = float(summary.get("percent_covered", 0.0))
        missing_lines = int(summary.get("missing_lines", 0))
        missing_branches = int(summary.get("missing_branches", 0))
        partial_branches = int(summary.get("num_partial_branches", 0))

        # Check function coverage
        funcs_undercovered = False
        for fn_data in file_data.get("functions", {}).values():
            fn_summary = fn_data.get("summary", {})
            if fn_summary.get("missing_lines", 0) > 0 or fn_summary.get("missing_branches", 0) > 0:
                funcs_undercovered = True
                break

        if pct < 100.0 or missing_lines > 0 or missing_branches > 0 or partial_branches > 0 or funcs_undercovered:
            undercovered.append((norm_path, pct, missing_lines, missing_branches))

    # Add any source files completely missing from the test coverage extraction
    for f in sorted(files_in_src - covered_files):
        undercovered.append((f, 0.0, -1, -1))

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
