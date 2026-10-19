"""Unit tests verifying snapshot drift detection and changelog generation."""

from __future__ import annotations

import io
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_snapshot_drift import (
    check_directory_drift,
    compare_snapshot_files,
    load_snapshot_file,
    main,
    parse_arguments,
)


def test_load_snapshot_file_success(tmp_path: Path) -> None:
    """Test loading valid snapshot JSON file.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    file_path = tmp_path / "snap.json"
    file_path.write_text('{"version": "1.0.0", "categories": {}}', encoding="utf-8")
    loaded = load_snapshot_file(str(file_path))
    assert loaded["version"] == "1.0.0"


def test_load_snapshot_file_not_found(tmp_path: Path) -> None:
    """Test FileNotFoundError raised when snapshot file does not exist.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_snapshot_file(str(missing))


def test_compare_snapshot_files(tmp_path: Path) -> None:
    """Test comparing two snapshot files for differences.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    old_file.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "categories": {"math": [{"name": "add", "api_path": "torch.add", "params": [{"name": "a"}]}]},
            }
        ),
        encoding="utf-8",
    )
    new_file.write_text(
        json.dumps(
            {
                "version": "1.1.0",
                "categories": {"math": [{"name": "add", "api_path": "torch.add", "params": [{"name": "a"}, {"name": "b", "default": 0}]}]},
            }
        ),
        encoding="utf-8",
    )

    diff, changelog = compare_snapshot_files(str(old_file), str(new_file))
    assert len(diff.non_breaking_signature_changed) == 1
    assert "Changelog" in changelog


def test_check_directory_drift_nonexistent_paths() -> None:
    """Test check_directory_drift with invalid/missing directory paths."""
    res, breaking = check_directory_drift("/nonexistent/dir1", "/nonexistent/dir2")
    assert res == {}
    assert breaking == []


def test_check_directory_drift_with_breaking_changes(tmp_path: Path) -> None:
    """Test check_directory_drift detecting breaking signature changes and removals.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_dir = tmp_path / "old_dir"
    new_dir = tmp_path / "new_dir"
    old_dir.mkdir()
    new_dir.mkdir()

    # Create breaking signature change: required parameter added
    (old_dir / "torch.json").write_text(
        json.dumps(
            {
                "categories": {
                    "math": [
                        {"name": "sub", "api_path": "torch.sub", "params": [{"name": "a"}]},
                        {"name": "mul", "api_path": "torch.mul", "params": [{"name": "a"}]},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (new_dir / "torch.json").write_text(
        json.dumps(
            {
                "categories": {
                    "math": [
                        # sub has required parameter added (breaking)
                        {"name": "sub", "api_path": "torch.sub", "params": [{"name": "a"}, {"name": "b"}]},
                        # mul was removed (breaking)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    results, breaking = check_directory_drift(str(old_dir), str(new_dir))
    assert "torch.json" in results
    assert len(breaking) >= 2


def test_parse_arguments() -> None:
    """Test argument parsing for snapshot drift tool."""
    ns = parse_arguments(["--old", "old_path", "--new", "new_path", "--allow-breaking"])
    assert ns.old == "old_path"
    assert ns.new == "new_path"
    assert ns.allow_breaking is True
    assert ns.output_changelog is None


def test_main_files_non_breaking(tmp_path: Path) -> None:
    """Test main execution comparing two files with non-breaking changes.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    out_log = tmp_path / "changelog.md"

    snap_content = json.dumps({"version": "1.0", "categories": {}})
    old_file.write_text(snap_content, encoding="utf-8")
    new_file.write_text(snap_content, encoding="utf-8")

    buf = io.StringIO()
    exit_code = main(
        [
            "--old",
            str(old_file),
            "--new",
            str(new_file),
            "--output-changelog",
            str(out_log),
        ],
        stdout=buf,
    )
    assert exit_code == 0
    assert out_log.exists()


def test_main_files_breaking_changes(tmp_path: Path) -> None:
    """Test main execution with breaking changes rejected and allowed.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"

    old_file.write_text(
        json.dumps({"categories": {"math": [{"name": "op", "api_path": "torch.op", "params": []}]}}),
        encoding="utf-8",
    )
    new_file.write_text(
        json.dumps({"categories": {"math": []}}),
        encoding="utf-8",
    )

    buf = io.StringIO()
    # Reject breaking changes by default
    exit_code = main(["--old", str(old_file), "--new", str(new_file)], stdout=buf)
    assert exit_code == 1

    # Allow breaking changes with flag
    buf_allowed = io.StringIO()
    exit_code_allowed = main(
        ["--old", str(old_file), "--new", str(new_file), "--allow-breaking"],
        stdout=buf_allowed,
    )
    assert exit_code_allowed == 0


def test_main_directories_success(tmp_path: Path) -> None:
    """Test main execution comparing directories without breaking changes.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_dir = tmp_path / "dir1"
    new_dir = tmp_path / "dir2"
    old_dir.mkdir()
    new_dir.mkdir()

    content = json.dumps({"categories": {}})
    (old_dir / "target.json").write_text(content, encoding="utf-8")
    (new_dir / "target.json").write_text(content, encoding="utf-8")

    buf = io.StringIO()
    exit_code = main(["--old", str(old_dir), "--new", str(new_dir)], stdout=buf)
    assert exit_code == 0
    assert "Directory snapshot drift check passed" in buf.getvalue()


def test_main_directories_breaking_changes(tmp_path: Path) -> None:
    """Test main execution comparing directories with breaking changes.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_dir = tmp_path / "dir1"
    new_dir = tmp_path / "dir2"
    old_dir.mkdir()
    new_dir.mkdir()

    (old_dir / "target.json").write_text(
        json.dumps({"categories": {"math": [{"name": "op", "api_path": "torch.op", "params": []}]}}),
        encoding="utf-8",
    )
    (new_dir / "target.json").write_text(
        json.dumps({"categories": {"math": []}}),
        encoding="utf-8",
    )

    buf = io.StringIO()
    exit_code = main(["--old", str(old_dir), "--new", str(new_dir)], stdout=buf)
    assert exit_code == 1

    buf_allowed = io.StringIO()
    exit_code_allowed = main(
        ["--old", str(old_dir), "--new", str(new_dir), "--allow-breaking"],
        stdout=buf_allowed,
    )
    assert exit_code_allowed == 0


def test_main_mixed_paths_error(tmp_path: Path) -> None:
    """Test error returned when mixing a file and a directory path.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    file_path = tmp_path / "file.json"
    file_path.write_text("{}", encoding="utf-8")
    dir_path = tmp_path / "dir"
    dir_path.mkdir()

    buf = io.StringIO()
    exit_code = main(["--old", str(file_path), "--new", str(dir_path)], stdout=buf)
    assert exit_code == 1
    assert "Error: Both --old and --new must be either files or directories" in buf.getvalue()


def test_main_framework_snapshots_resolution(tmp_path: Path) -> None:
    """Test resolving ../ml-framework-snapshots and snapshots/ subdirectories.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    buf = io.StringIO()
    exit_code = main(["--old", "../ml-framework-snapshots", "--new", "../ml-framework-snapshots"], stdout=buf)
    assert exit_code == 0
    assert "Directory snapshot drift check passed" in buf.getvalue()

    parent_dir = tmp_path / "repo_root"
    snap_sub = parent_dir / "snapshots"
    snap_sub.mkdir(parents=True)
    (snap_sub / "target.json").write_text(json.dumps({"categories": {}}), encoding="utf-8")

    buf_sub = io.StringIO()
    exit_sub = main(["--old", str(parent_dir), "--new", str(parent_dir)], stdout=buf_sub)
    assert exit_sub == 0

    buf_missing = io.StringIO()
    exit_missing = main(["--old", str(tmp_path / "does_not_exist")], stdout=buf_missing)
    assert exit_missing == 1
    assert "Error: Snapshot path not found" in buf_missing.getvalue()


def test_check_directory_drift_missing_default_target(tmp_path: Path) -> None:
    """Test check_directory_drift reporting a missing target on the default snapshot directory.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    from ml_switcheroo_compiler.backends.snapshot_grounding import (
        BackendSnapshotTargetModel,
        BackendSnapshotTargetsConfig,
    )

    fake_cfg = BackendSnapshotTargetsConfig(
        targets={
            "nonexistent_backend": BackendSnapshotTargetModel(
                framework="nonexistent_fw",
                snapshot_glob="nonexistent_v*.json",
                status="available",
            )
        }
    )
    with (
        patch("scripts.check_snapshot_drift._resolve_default_snapshot_dir", return_value=str(tmp_path)),
        patch("scripts.check_snapshot_drift.SnapshotGroundingEngine.get_snapshot_path", return_value=None),
        patch("ml_switcheroo_compiler.backends.snapshot_grounding.load_backend_snapshot_targets", return_value=fake_cfg),
    ):
        _, breaking = check_directory_drift(str(tmp_path), str(tmp_path))
        assert any("nonexistent_backend" in b for b in breaking)


def test_main_block_execution(tmp_path: Path) -> None:
    """Test __main__ module execution via runpy.

    Args:
        tmp_path (Path): Temporary test directory path.
    """
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    content = json.dumps({"categories": {}})
    old_file.write_text(content, encoding="utf-8")
    new_file.write_text(content, encoding="utf-8")

    test_args = ["scripts.check_snapshot_drift", "--old", str(old_file), "--new", str(new_file)]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.check_snapshot_drift", run_name="__main__")
        assert exc_info.value.code == 0


def test_is_breaking_and_flatten_edge_cases() -> None:
    """Test _is_breaking_param_change and _flatten_snapshot_items edge branches."""
    from scripts.check_snapshot_drift import (
        DiffResult,
        _flatten_snapshot_items,
        _is_breaking_param_change,
        diff_snapshots,
        generate_changelog,
    )

    assert _is_breaking_param_change([{"name": "x"}, {"name": "y"}], [{"name": "x"}]) is True
    assert _is_breaking_param_change([{"name": "x"}], [{"name": "x"}, {"name": "y", "default": None}]) is True
    assert _is_breaking_param_change([{"name": "x", "default": "1"}], [{"name": "x", "default": None}]) is True
    assert _is_breaking_param_change([{"name": "x", "default": "1"}], [{"name": "x", "default": "2"}]) is True
    assert _is_breaking_param_change([{"name": "x", "default": "1"}], [{"name": "x", "default": "1"}, {"name": "args", "kind": "VAR_POSITIONAL"}]) is False

    res_non_break = diff_snapshots(
        {"categories": {"math": [{"api_path": "f", "params": [{"name": "x", "annotation": "int"}]}, {"api_path": "same", "params": [{"name": "x"}]}]}},
        {"categories": {"math": [{"api_path": "f", "params": [{"name": "x", "annotation": "float"}]}, {"api_path": "same", "params": [{"name": "x"}]}, {"api_path": "added_fn", "params": []}]}},
    )
    assert res_non_break.non_breaking_signature_changed == ["f"]
    assert res_non_break.added == ["added_fn"]
    assert "Non-Breaking Signature Changes" in generate_changelog(res_non_break)

    assert _flatten_snapshot_items({"categories": "not_a_dict"}) == {}
    assert _flatten_snapshot_items({"categories": {"math": "not_a_list"}}) == {}
    assert _flatten_snapshot_items({"categories": {"math": ["not_a_dict", {"name": ""}]}}) == {}

    empty_changelog = generate_changelog(DiffResult())
    assert "No changes detected." in empty_changelog
