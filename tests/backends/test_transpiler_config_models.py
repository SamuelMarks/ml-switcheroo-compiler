"""Tests for transpiler config models."""

from pathlib import Path

from ml_switcheroo_compiler.backends.transpiler_config_models import FrameworkConfig, TranspilerConfig, load_transpiler_config


def test_framework_config() -> None:
    """Test FrameworkConfig."""
    config = FrameworkConfig(target_module="jax", module_path=["jax", "numpy"])
    assert config.target_module == "jax"
    assert config.module_path == ["jax", "numpy"]
    assert config.kwarg_map == {}
    assert config.class_bases == {}
    assert config.method_map == {}
    assert config.broadcast_method == "broadcast_to"


def test_transpiler_config() -> None:
    """Test TranspilerConfig."""
    fconfig = FrameworkConfig(target_module="jax", module_path=["jax", "numpy"])
    tconfig = TranspilerConfig(frameworks={"jax": fconfig})
    assert tconfig.frameworks["jax"].target_module == "jax"


def test_load_transpiler_config(tmp_path: Path) -> None:
    """Test loading transpiler config."""
    yaml_content = """
frameworks:
  jax:
    target_module: "jax"
    module_path: ["jax", "numpy"]
"""
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as f:
        f.write(yaml_content)

    config = load_transpiler_config(str(file_path))
    assert "jax" in config.frameworks
    assert config.frameworks["jax"].target_module == "jax"


def test_is_callable_op_symbol() -> None:
    """Test _is_callable_op_symbol filtering."""
    from ml_switcheroo_compiler.backends.transpiler_config_models import _is_callable_op_symbol

    assert _is_callable_op_symbol("torch.add") is True
    assert _is_callable_op_symbol("torch.acceleratorerror") is False
    assert _is_callable_op_symbol("torch.cuda.OutOfMemoryError") is False
    assert _is_callable_op_symbol("torch.UserWarning") is False
    assert _is_callable_op_symbol("torch._internal_helper") is False
    assert _is_callable_op_symbol("CharacterType") is False


def test_load_transpiler_config_empty(tmp_path: Path) -> None:
    """Test loading empty or non-dict transpiler config."""
    file_path = tmp_path / "empty.yaml"
    with open(file_path, "w") as f:
        f.write("")

    config = load_transpiler_config(str(file_path))
    assert config.frameworks == {}

    # Non-dict YAML (e.g. list)
    list_file = tmp_path / "list.yaml"
    with open(list_file, "w") as f:
        f.write("- item1\n- item2\n")
    config2 = load_transpiler_config(str(list_file))
    assert config2.frameworks == {}


def test_load_transpiler_config_rules_dir(tmp_path: Path) -> None:
    """Test loading from transpiler rules directory."""
    import os
    from unittest.mock import patch

    cfg_file = tmp_path / "main.yaml"
    with open(cfg_file, "w") as f:
        f.write("frameworks: {}\n")

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "ignored.txt").write_text("not yaml")
    (rules_dir / "valid.yaml").write_text('ast_to_ir_ops:\n  torch.mul: Mul\nir_to_ast_ops:\n  Mul:\n    torch: ["torch.mul"]\n')
    (rules_dir / "only_ast.yaml").write_text("ast_to_ir_ops:\n  torch.sin: Sin\n")
    (rules_dir / "only_ir.yaml").write_text("ir_to_ast_ops:\n  Cos:\n    torch: ['torch.cos']\n")
    (rules_dir / "non_dict.yaml").write_text("- not a dict\n")
    (rules_dir / "broken.yaml").write_text(":\nbroken yaml")

    real_join = os.path.join
    real_isdir = os.path.isdir
    real_listdir = os.listdir

    def mock_join(*args: str) -> str:
        if "rules" in args:
            if args[-1] == "rules":
                return str(rules_dir)
            return str(rules_dir / args[-1])
        return real_join(*args)

    def mock_isdir(path: str) -> bool:
        if str(rules_dir) in str(path) or str(path).endswith("rules"):
            return True
        return real_isdir(path)

    def mock_listdir(path: str) -> list[str]:
        if str(rules_dir) in str(path) or str(path).endswith("rules"):
            return ["ignored.txt", "valid.yaml", "only_ast.yaml", "only_ir.yaml", "non_dict.yaml", "broken.yaml"]
        return real_listdir(path)

    with patch("os.path.isdir", side_effect=mock_isdir), patch("os.listdir", side_effect=mock_listdir), patch("os.path.join", side_effect=mock_join):
        cfg = load_transpiler_config(str(cfg_file))
        assert "torch.mul" in cfg.ast_to_ir_ops
        assert "torch.sin" in cfg.ast_to_ir_ops
        assert "Mul" in cfg.ir_to_ast_ops
        assert "Cos" in cfg.ir_to_ast_ops

    def mock_isdir_false(path: str) -> bool:
        if str(rules_dir) in str(path) or "rules" in str(path):
            return False
        return real_isdir(path)

    # Test when rules_dir does not exist (branch 98->114)
    with patch("os.path.isdir", side_effect=mock_isdir_false):
        cfg_no_dir = load_transpiler_config(str(cfg_file))
        assert cfg_no_dir.frameworks == {}


def test_load_method_rewrites_coverage() -> None:
    """Test _load_method_rewrites handling of invalid file and exceptions."""
    from unittest.mock import mock_open, patch

    from ml_switcheroo_compiler.backends.transpiler_config_models import (
        SyntacticPatternsConfig,
        _load_method_rewrites,
    )

    patterns = SyntacticPatternsConfig()

    # When file does not exist
    with patch("os.path.isfile", return_value=False):
        _load_method_rewrites(patterns)
        assert len(patterns.method_to_function_rewrites) == 0

    # When file raises exception on open
    with patch("os.path.isfile", return_value=True), patch("builtins.open", side_effect=OSError("Read failure")):
        _load_method_rewrites(patterns)
        assert len(patterns.method_to_function_rewrites) == 0

    # When file contains valid content
    valid_yaml = "method_to_function_rewrites:\n  - pattern: 'a.b()'\n    replacement: 'b(a)'\n"
    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=valid_yaml)):
        _load_method_rewrites(patterns)
        assert len(patterns.method_to_function_rewrites) > 0

    # When file contains dict without the rewrite key
    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data="other: 123\n")):
        _load_method_rewrites(patterns)


def test_syntactic_patterns_models() -> None:
    """Test ASTPatternRule, StateLiftingPattern, and SyntacticPatternsConfig models."""
    from ml_switcheroo_compiler.backends.transpiler_config_models import (
        ASTPatternRule,
        StateLiftingPattern,
        SyntacticPatternsConfig,
    )

    rule = ASTPatternRule(
        pattern="narrow",
        replacement="slice",
        description="narrow to slice",
        source_framework="pytorch",
        target_framework="jax",
    )
    assert rule.pattern == "narrow"

    lifting = StateLiftingPattern(
        module_class="nn.Linear",
        param_names=["weight", "bias"],
        functional_op="Linear",
        init_mapping={"weight": "init_w", "bias": "init_b"},
    )
    assert lifting.module_class == "nn.Linear"
    assert lifting.param_names == ["weight", "bias"]

    cfg = SyntacticPatternsConfig(
        state_lifting_patterns=[lifting],
        tensor_creation_rewrites=[rule],
        slicing_rewrites=[rule],
        method_to_function_rewrites=[rule],
    )
    assert len(cfg.state_lifting_patterns) == 1
    assert len(cfg.tensor_creation_rewrites) == 1


def test_cst_rewrite_rules_models_and_loader() -> None:
    """Test loading and validating declarative CST rewrite rules."""
    from ml_switcheroo_compiler.backends.transpiler_config_models import (
        ArgumentPermutationSpec,
        ASTPatternMatchSpec,
        ASTRewriteReplacementSpec,
        CSTRewriteRuleEntry,
        load_cst_rewrite_rules,
    )

    # 1. Test bundled loader
    config = load_cst_rewrite_rules()
    assert len(config.method_substitutions) >= 1
    assert any(rule.match.pattern == "view" for rule in config.method_substitutions)
    assert any(rule.match.pattern == "contiguous" for rule in config.chain_removals)
    assert any(rule.permutation.source_arg == "dim" for rule in config.argument_permutations if rule.permutation)

    # Test with explicit path
    import os

    explicit_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends", "cst_rewrite_rules.yaml")
    if os.path.exists(explicit_path):
        config_explicit = load_cst_rewrite_rules(path=explicit_path)
        assert len(config_explicit.method_substitutions) >= 1

    # 2. Test model validation
    match_spec = ASTPatternMatchSpec(pattern="test_pat", source_framework="pytorch")
    assert match_spec.pattern == "test_pat"

    repl_spec = ASTRewriteReplacementSpec(target_name="test_target", target_framework="jax")
    assert repl_spec.target_name == "test_target"

    perm_spec = ArgumentPermutationSpec(source_arg="dim", target_arg="axis")
    assert perm_spec.source_arg == "dim"

    entry = CSTRewriteRuleEntry(match=match_spec, replacement=repl_spec, permutation=perm_spec, description="test rule")
    assert entry.description == "test rule"
