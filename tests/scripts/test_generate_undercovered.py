"""Tests for scripts/generate_undercovered.py."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.generate_undercovered as gu


def test_is_file_undercovered() -> None:
    """Test _is_file_undercovered across covered and undercovered summaries and functions."""
    funcs_covered: dict[str, dict[str, gu.CoverageSummary]] = {
        "fn1": {
            "summary": {
                "missing_lines": 0,
                "missing_branches": 0,
            }
        }
    }
    covered_entry: gu.CoverageEntry = {
        "summary": {
            "percent_covered": 100.0,
            "missing_lines": 0,
            "missing_branches": 0,
            "num_partial_branches": 0,
        },
        "functions": funcs_covered,
    }
    assert gu._is_file_undercovered(covered_entry) == (False, 100.0, 0, 0)

    funcs_under: dict[str, dict[str, gu.CoverageSummary]] = {
        "fn_bad": {
            "summary": {
                "missing_lines": 0,
                "missing_branches": 1,
            }
        }
    }
    func_under_entry: gu.CoverageEntry = {
        "summary": {
            "percent_covered": 100.0,
            "missing_lines": 0,
            "missing_branches": 0,
            "num_partial_branches": 0,
        },
        "functions": funcs_under,
    }
    assert gu._is_file_undercovered(func_under_entry) == (True, 100.0, 0, 0)


def test_generate_undercovered_list_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_undercovered_list raising FileNotFoundError when coverage_extract.json is absent.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Coverage file not found"):
        gu.generate_undercovered_list()


def test_main_and_generate_undercovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full execution of generate_undercovered_list and main, including __main__ block.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "src" / "ml_switcheroo_compiler"
    src_dir.mkdir(parents=True)
    (src_dir / "covered.py").write_text("x = 1", encoding="utf-8")
    (src_dir / "undercovered.py").write_text("y = 2", encoding="utf-8")
    (src_dir / "unexecuted.py").write_text("z = 3", encoding="utf-8")
    (src_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    cov_data = {
        "files": {
            "scripts/other.py": {
                "summary": {"percent_covered": 50.0, "missing_lines": 5, "missing_branches": 1},
                "functions": {},
            },
            "src/ml_switcheroo_compiler/covered.py": {
                "summary": {
                    "percent_covered": 100.0,
                    "missing_lines": 0,
                    "missing_branches": 0,
                    "num_partial_branches": 0,
                },
                "functions": {},
            },
            "src/ml_switcheroo_compiler/undercovered.py": {
                "summary": {
                    "percent_covered": 80.0,
                    "missing_lines": 2,
                    "missing_branches": 1,
                    "num_partial_branches": 1,
                },
                "functions": {},
            },
        }
    }
    (tmp_path / "coverage_extract.json").write_text(json.dumps(cov_data), encoding="utf-8")

    gu.main()
    out_md = (tmp_path / "UNDERCOVERED.md").read_text(encoding="utf-8")
    assert "Total undercovered files: **2**" in out_md
    assert "undercovered.py" in out_md
    assert "80.0% coverage, 2 missing lines, 1 missing branches" in out_md
    assert "unexecuted.py" in out_md
    assert "0.0% coverage - no test execution" in out_md
    assert "`src/ml_switcheroo_compiler/covered.py`" not in out_md

    # Test empty undercovered list (branch 79->86)
    cov_data_all = {
        "files": {
            "src/ml_switcheroo_compiler/covered.py": {
                "summary": {"percent_covered": 100.0, "missing_lines": 0, "missing_branches": 0, "num_partial_branches": 0},
                "functions": {},
            },
            "src/ml_switcheroo_compiler/undercovered.py": {
                "summary": {"percent_covered": 100.0, "missing_lines": 0, "missing_branches": 0, "num_partial_branches": 0},
                "functions": {},
            },
            "src/ml_switcheroo_compiler/unexecuted.py": {
                "summary": {"percent_covered": 100.0, "missing_lines": 0, "missing_branches": 0, "num_partial_branches": 0},
                "functions": {},
            },
        }
    }
    (tmp_path / "coverage_extract.json").write_text(json.dumps(cov_data_all), encoding="utf-8")
    script_path = os.path.abspath(gu.__file__)
    with patch.object(sys, "argv", ["generate_undercovered.py"]):
        runpy.run_path(script_path, run_name="__main__")
    out_md_empty = (tmp_path / "UNDERCOVERED.md").read_text(encoding="utf-8")
    assert "Total undercovered files: **0**" in out_md_empty
