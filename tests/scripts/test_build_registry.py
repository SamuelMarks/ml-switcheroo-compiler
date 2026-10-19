"""Tests for the build_registry script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import mock_open, patch

import yaml

import scripts.build_registry as br

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_build_registry_multi_op(tmp_path: Path) -> None:
    """Test building registry with various file types and structures.

    Args:
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    def_dir = tmp_path / "definitions"
    def_dir.mkdir()
    out_file = tmp_path / "generated_registry.py"

    # 1. Normal multi-op dict
    data_multi: dict[str, dict[str, str]] = {"DummyOp1": {"signature": "(x) -> x"}, "DummyOp2": {"signature": "(y) -> y"}}
    with open(def_dir / "a_multi.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_multi, f)

    # 2. Single-op file with "operation" key and aliases
    data_single = {"operation": "DummyOp3", "signature": "(z) -> z", "aliases": ["DummyAlias1"]}
    with open(def_dir / "b_single.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_single, f)

    # 2b. Single-op file without aliases key
    data_no_aliases = {"operation": "DummyOpNoAliases", "signature": "(n) -> n"}
    with open(def_dir / "b_no_aliases.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_no_aliases, f)

    # 3. File with non-dict inner data to test branch `if isinstance(op_info, dict):`
    data_invalid = {"DummyOp4": "not a dict"}
    with open(def_dir / "c_invalid.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_invalid, f)

    # 3b. File with root non-dict data
    with open(def_dir / "c_list.yaml", "w", encoding="utf-8") as f:
        f.write("- item1\n- item2\n")

    # 4. Non-yaml file to test branch `if filename.endswith(".yaml"):`
    with open(def_dir / "d_not_yaml.txt", "w", encoding="utf-8") as f:
        f.write("ignore me")

    br.build(definitions_dir=str(def_dir), out_file=str(out_file))

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")

    # Check outputs
    assert "DummyOp1" in content
    assert "DummyOp2" in content
    assert "DummyOp3" in content
    assert "DummyOp4" not in content


def _test_load_backend_mappings_and_normalization_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_load_backend_mappings_and_normalization.

    Args:
        _state: Shared test state dictionary.
    """
    tmp_path = _state["tmp_path"]
    empty_maps = cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(br, "load_backend_mappings")))(cast("EagerCallableProtocol", str)(tmp_path / "nonexistent"))
    assert empty_maps == {}

    # Setup dummy backends directory
    backends_dir = tmp_path / "backends"
    backends_dir.mkdir()
    (backends_dir / "not_a_dir.txt").write_text("ignore", encoding="utf-8")
    (backends_dir / "no_map_dir").mkdir()

    numpy_map_dir = backends_dir / "numpy" / "mappings"
    numpy_map_dir.mkdir(parents=True)
    (numpy_map_dir / "Add.yaml").write_text("Add:\n  target_api: np.add\n", encoding="utf-8")
    (numpy_map_dir / "list.yaml").write_text("- not_a_dict_mapping\n", encoding="utf-8")
    (numpy_map_dir / "no_target.yaml").write_text("NoTarget:\n  other: 123\n", encoding="utf-8")
    (numpy_map_dir / "not_dict_entry.yaml").write_text("NotDict: 123\n", encoding="utf-8")
    (numpy_map_dir / "invalid.yaml").write_text("invalid_yaml: [:\n", encoding="utf-8")
    (numpy_map_dir / "not_yaml.txt").write_text("ignore", encoding="utf-8")
    (numpy_map_dir / "z_another.yaml").write_text("Z:\n  target_api: np.z\n", encoding="utf-8")

    loaded_maps = cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(br, "load_backend_mappings")))(cast("EagerCallableProtocol", str)(backends_dir))
    assert "numpy" in loaded_maps
    assert loaded_maps["numpy"].get("Add") == "np.add"

    # Test normalization of ops with various variant combinations
    op_add = {
        "operation": "Add",
        "variants": {
            "numpy": {},
            "cupy": {},
            "llvm_cpp": {"template": "binary", "scalar_expr": "in0_val"},
            "custom": {"generator": "emit_custom"},
        },
    }
    norm_add = cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(br, "_normalize_and_validate_op")))("Add", op_add, loaded_maps)
    variants = norm_add["variants"]
    assert isinstance(variants, dict)
    getattr(_state, "update")({"backends_dir": backends_dir, "empty_maps": empty_maps, "loaded_maps": loaded_maps, "norm_add": norm_add})
    getattr(_state, "update")({"numpy_map_dir": numpy_map_dir, "op_add": op_add, "variants": variants})
    _state.update(locals())


def _test_load_backend_mappings_and_normalization_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_load_backend_mappings_and_normalization.

    Args:
        _state: Shared test state dictionary.
    """
    loaded_maps, op_add, tmp_path, variants = _state["loaded_maps"], _state["op_add"], _state["tmp_path"], _state["variants"]
    assert variants["numpy"] == {"target_api": "np.add", "supported": True}
    assert variants["cupy"] == {"supported": False}
    assert variants["llvm_cpp"]["scalar_expr"] == "in0_val + in1_val"
    assert bool(variants["llvm_cpp"]["supported"]) is True
    assert bool(variants["custom"]["supported"]) is True

    # Test normalization of op with unsupported llvm_cpp fallback
    op_unknown = {
        "operation": "UnknownOp",
        "variants": {
            "llvm_cpp": {"template": "unary", "scalar_expr": "in0_val"},
        },
        "math_semantics": {"differentiable": False},
    }
    norm_unk = cast("EagerCallableProtocol", getattr(br, "_normalize_and_validate_op"))("UnknownOp", op_unknown, loaded_maps)
    unk_variants = norm_unk["variants"]
    assert isinstance(unk_variants, dict)
    assert unk_variants["llvm_cpp"] == {"supported": False}
    assert norm_unk["math_semantics"] == {"differentiable": False}

    # Test normalization of Identity op preserving in0_val
    op_ident = {
        "operation": "Identity",
        "variants": {
            "llvm_cpp": {"template": "unary", "scalar_expr": "in0_val"},
        },
    }
    norm_ident = cast("EagerCallableProtocol", getattr(br, "_normalize_and_validate_op"))("Identity", op_ident, loaded_maps)
    ident_variants = norm_ident["variants"]
    assert isinstance(ident_variants, dict)
    assert cast("EagerTensorProtocol", cast("EagerTensorProtocol", ident_variants)["llvm_cpp"])["scalar_expr"] == "in0_val"
    assert cast("EagerTensorProtocol", cast("EagerTensorProtocol", ident_variants)["llvm_cpp"])["supported"] is True

    # Test custom out_yaml build
    def_dir = tmp_path / "definitions"
    def_dir.mkdir()
    with open(def_dir / "Add.yaml", "w", encoding="utf-8") as f:
        yaml.dump(op_add, f)

    out_file = tmp_path / "custom_gen.py"
    getattr(_state, "update")({"def_dir": def_dir, "f": f, "ident_variants": ident_variants, "norm_ident": norm_ident})
    getattr(_state, "update")({"norm_unk": norm_unk, "op_ident": op_ident, "op_unknown": op_unknown, "out_file": out_file})
    getattr(_state, "update")({"unk_variants": unk_variants})
    _state.update(locals())


def _test_load_backend_mappings_and_normalization_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_load_backend_mappings_and_normalization.

    Args:
        _state: Shared test state dictionary.
    """
    backends_dir, def_dir, out_file, tmp_path = _state["backends_dir"], _state["def_dir"], _state["out_file"], _state["tmp_path"]
    out_yaml = tmp_path / "custom_ops.yaml"
    br.build(
        definitions_dir=str(def_dir),
        out_file=str(out_file),
        out_yaml=str(out_yaml),
        backends_dir=str(backends_dir),
    )
    assert out_file.exists()
    assert out_yaml.exists()

    # Test out_yaml is None and out_file not ending with generated_registry.py (branch 243->exit)
    out_file_no_yaml = tmp_path / "other_registry_output.txt"
    br.build(
        definitions_dir=str(def_dir),
        out_file=str(out_file_no_yaml),
        out_yaml=None,
        backends_dir=str(backends_dir),
    )
    assert out_file_no_yaml.exists()
    getattr(_state, "update")({"out_file_no_yaml": out_file_no_yaml, "out_yaml": out_yaml})


def test_load_backend_mappings_and_normalization(tmp_path: Path) -> None:
    """Test loading backend mappings, variant normalization, and yaml output.

    Args:
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    _state: dict[str, EagerTensorProtocol] = cast("dict[str, EagerTensorProtocol]", {"tmp_path": tmp_path})
    _test_load_backend_mappings_and_normalization_part_1(_state)
    _test_load_backend_mappings_and_normalization_part_2(_state)
    _test_load_backend_mappings_and_normalization_part_3(_state)
    assert len(_state) > 0


def test_build_registry_main(tmp_path: Path) -> None:
    """Test the __main__ block of build_registry.py.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    with patch("scripts.build_registry.build") as mock_build:
        br.main()
        mock_build.assert_called_once()

    # To cover `if __name__ == "__main__":`
    # run_path recompiles the file, so we can't patch build() from the module.
    # We let it run harmlessly by giving it empty directories.
    def_dir = tmp_path / "definitions"
    def_dir.mkdir()
    tmp_path / "generated.py"

    with patch.object(sys, "argv", ["build_registry.py"]):
        # We patch os.listdir to return empty, so it does no work.
        sys.modules.pop("scripts.build_registry", None)
        with patch("os.listdir", return_value=[]):
            with patch("builtins.open", mock_open()):
                runpy.run_module("scripts.build_registry", run_name="__main__")
