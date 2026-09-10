"""Pydantic models for transpilation rules."""

from __future__ import annotations

import os

import yaml
from pydantic import BaseModel, Field


class FrameworkConfig(BaseModel):
    """Configuration for a specific framework."""

    target_module: str
    module_path: list[str]
    kwarg_map: dict[str, str] = Field(default_factory=dict)
    class_bases: dict[str, list[str]] = Field(default_factory=dict)
    method_map: dict[str, str] = Field(default_factory=dict)
    broadcast_method: str = "broadcast_to"


class TranspilationFrameworkConfig(FrameworkConfig):
    """Declarative Pydantic schema for target framework code emission and transpilation rules."""

    framework_name: str | None = None


class ASTPatternRule(BaseModel):
    """Declarative AST match-and-replace pattern rule."""

    pattern: str
    replacement: str
    description: str | None = None
    source_framework: str | None = None
    target_framework: str | None = None


class StateLiftingPattern(BaseModel):
    """Declarative pattern for lifting OOP state into functional parameters."""

    module_class: str
    param_names: list[str]
    functional_op: str
    init_mapping: dict[str, str] = Field(default_factory=dict)


class SyntacticPatternsConfig(BaseModel):
    """Configuration schema for declarative syntactic AST transformation patterns."""

    state_lifting_patterns: list[StateLiftingPattern] = Field(default_factory=list)
    tensor_creation_rewrites: list[ASTPatternRule] = Field(default_factory=list)
    slicing_rewrites: list[ASTPatternRule] = Field(default_factory=list)
    method_to_function_rewrites: list[ASTPatternRule] = Field(default_factory=list)


class TranspilerConfig(BaseModel):
    """Root configuration for transpiler."""

    frameworks: dict[str, FrameworkConfig]
    ast_to_ir_ops: dict[str, str] = Field(default_factory=dict)
    ir_to_ast_ops: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    syntactic_patterns: SyntacticPatternsConfig = Field(default_factory=SyntacticPatternsConfig)


def _is_callable_op_symbol(symbol: str) -> bool:
    """Check if an AST symbol represents a callable operation rather than an exception or internal symbol.

    Args:
        symbol (str): Dot-separated symbol identifier.

    Returns:
        bool: True if callable operation, False if exception or private helper.
    """
    sym_lower = symbol.lower()
    last_part = symbol.split(".")[-1]
    if "error" in sym_lower or "exception" in sym_lower or "warning" in sym_lower or last_part.startswith("_") or last_part.lower() == "charactertype":
        return False
    return True


def _load_domain_manifests(
    ast_to_ir_ops: dict[str, str],
    ir_to_ast_ops: dict[str, dict[str, list[str]]],
) -> None:
    """Load AST to IR and IR to AST operations from domain rule manifests.

    Args:
        ast_to_ir_ops (dict[str, str]): Output map from source AST symbols to IR ops.
        ir_to_ast_ops (dict[str, dict[str, list[str]]]): Output map from IR ops to framework AST nodes.
    """
    rules_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transpiler", "rules")
    if not os.path.isdir(rules_dir):
        return

    for fname in sorted(os.listdir(rules_dir)):
        if not fname.endswith(".yaml"):
            continue
        fpath: str = os.path.join(rules_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as rf:
                rdata: dict[str, object] = yaml.safe_load(rf) or {}
                if isinstance(rdata, dict):
                    if "ast_to_ir_ops" in rdata and isinstance(rdata["ast_to_ir_ops"], dict):
                        ast_to_ir_ops.update(rdata["ast_to_ir_ops"])
                    if "ir_to_ast_ops" in rdata and isinstance(rdata["ir_to_ast_ops"], dict):
                        ir_to_ast_ops.update(rdata["ir_to_ast_ops"])
        except Exception:
            pass


def _load_framework_configs(frameworks: dict[str, FrameworkConfig]) -> None:
    """Load modular framework target configurations from YAML manifests.

    Args:
        frameworks (dict[str, FrameworkConfig]): Framework mapping to populate.
    """
    frameworks_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules", "frameworks")
    if not os.path.isdir(frameworks_dir):
        return

    for fname in sorted(os.listdir(frameworks_dir)):
        if not fname.endswith(".yaml"):
            continue
        fw_name: str = fname[:-5]
        fpath: str = os.path.join(frameworks_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as rf:
                fw_data: dict[str, object] = yaml.safe_load(rf) or {}
                if isinstance(fw_data, dict) and fw_name not in frameworks:
                    fw_cfg = TranspilationFrameworkConfig.model_validate(fw_data)
                    fw_cfg.framework_name = fw_name
                    frameworks[fw_name] = fw_cfg
        except Exception:
            pass


def _load_method_rewrites(syntactic_patterns: SyntacticPatternsConfig) -> None:
    """Load declarative method-to-function AST rewrite rules.

    Args:
        syntactic_patterns (SyntacticPatternsConfig): Pattern configuration to populate.
    """
    method_rewrites_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules", "method_rewrites.yaml")
    if not os.path.isfile(method_rewrites_path):
        return

    try:
        with open(method_rewrites_path, encoding="utf-8") as rf:
            mdata: dict[str, object] = yaml.safe_load(rf) or {}
            if isinstance(mdata, dict) and "method_to_function_rewrites" in mdata and isinstance(mdata["method_to_function_rewrites"], list):
                for item in mdata["method_to_function_rewrites"]:
                    syntactic_patterns.method_to_function_rewrites.append(ASTPatternRule.model_validate(item))
    except Exception:
        pass


def load_transpiler_config(yaml_path: str) -> TranspilerConfig:
    """Load transpiler configuration from YAML file and domain manifests.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        TranspilerConfig: The loaded configuration.
    """
    with open(yaml_path, encoding="utf-8") as f:
        data: dict[str, object] = yaml.safe_load(f) or {}

    if not isinstance(data, dict) or not data:
        return TranspilerConfig(
            frameworks={},
            ast_to_ir_ops={},
            ir_to_ast_ops={},
            syntactic_patterns=SyntacticPatternsConfig(),
        )

    frameworks: dict[str, FrameworkConfig] = data.get("frameworks", {}) or {}
    ast_to_ir_ops: dict[str, str] = dict(data.get("ast_to_ir_ops", {}))
    ir_to_ast_ops: dict[str, dict[str, list[str]]] = dict(data.get("ir_to_ast_ops", {}))
    raw_patterns = data.get("syntactic_patterns", {})
    syntactic_patterns: SyntacticPatternsConfig = SyntacticPatternsConfig.model_validate(raw_patterns) if isinstance(raw_patterns, dict) else SyntacticPatternsConfig()

    _load_domain_manifests(ast_to_ir_ops, ir_to_ast_ops)
    _load_framework_configs(frameworks)
    _load_method_rewrites(syntactic_patterns)

    clean_ast_to_ir: dict[str, str] = {k: v for k, v in ast_to_ir_ops.items() if _is_callable_op_symbol(k)}

    return TranspilerConfig(
        frameworks=frameworks,
        ast_to_ir_ops=clean_ast_to_ir,
        ir_to_ast_ops=ir_to_ast_ops,
        syntactic_patterns=syntactic_patterns,
    )


class ASTPatternMatchSpec(BaseModel):
    """Specification for matching an AST pattern.

    Attributes:
        pattern (str): Name or expression pattern to match.
        source_framework (str | None): Source framework constraint ('pytorch', 'jax', etc.).
    """

    pattern: str
    source_framework: str | None = None


class ASTRewriteReplacementSpec(BaseModel):
    """Specification for replacing a matched AST node.

    Attributes:
        target_name (str): Replacement method or function name.
        target_framework (str | None): Target framework constraint.
    """

    target_name: str
    target_framework: str | None = None


class ArgumentPermutationSpec(BaseModel):
    """Specification for argument permutations in call sites.

    Attributes:
        source_arg (str): Original argument name.
        target_arg (str): Renamed or mapped argument name.
    """

    source_arg: str
    target_arg: str


class BroadcastingRuleEntry(BaseModel):
    """Declarative broadcasting rewrite rule entry.

    Attributes:
        source_method (str): Source method name (e.g. 'expand').
        target_method (str): Target method name (e.g. 'broadcast_to').
        source_framework (str | None): Source framework constraint.
        target_framework (str | None): Target framework constraint.
    """

    source_method: str
    target_method: str
    source_framework: str | None = None
    target_framework: str | None = None


class CSTRewriteRuleEntry(BaseModel):
    """Declarative CST rewrite rule entry.

    Attributes:
        match (ASTPatternMatchSpec): Pattern match specification.
        replacement (ASTRewriteReplacementSpec | None): Replacement specification.
        permutation (ArgumentPermutationSpec | None): Argument permutation specification.
        description (str | None): Rule explanation.
    """

    match: ASTPatternMatchSpec
    replacement: ASTRewriteReplacementSpec | None = None
    permutation: ArgumentPermutationSpec | None = None
    description: str | None = None


class CSTRewriteRulesConfig(BaseModel):
    """Container for declarative CST rewriting rules.

    Attributes:
        method_substitutions (list[CSTRewriteRuleEntry]): Method call rename rules.
        chain_removals (list[CSTRewriteRuleEntry]): Chained call strip rules.
        argument_permutations (list[CSTRewriteRuleEntry]): Argument keyword rename rules.
        in_place_mutations (list[CSTRewriteRuleEntry]): In-place mutation functionalization rules.
        tensor_creation_rewrites (list[CSTRewriteRuleEntry]): Factory creation rewrite rules.
        slice_normalizations (list[CSTRewriteRuleEntry]): Slicing and indexing normalization rules.
        broadcasting_rules (list[BroadcastingRuleEntry]): Explicit broadcasting conversion rules.
    """

    method_substitutions: list[CSTRewriteRuleEntry] = Field(default_factory=list)
    chain_removals: list[CSTRewriteRuleEntry] = Field(default_factory=list)
    argument_permutations: list[CSTRewriteRuleEntry] = Field(default_factory=list)
    in_place_mutations: list[CSTRewriteRuleEntry] = Field(default_factory=list)
    tensor_creation_rewrites: list[CSTRewriteRuleEntry] = Field(default_factory=list)
    slice_normalizations: list[CSTRewriteRuleEntry] = Field(default_factory=list)
    broadcasting_rules: list[BroadcastingRuleEntry] = Field(default_factory=list)


class CSTRewriteRulesRootConfig(BaseModel):
    """Root model for cst_rewrite_rules.yaml.

    Attributes:
        cst_rewrite_rules (CSTRewriteRulesConfig): The rules configuration.
    """

    cst_rewrite_rules: CSTRewriteRulesConfig


def load_cst_rewrite_rules(path: str | None = None) -> CSTRewriteRulesConfig:
    """Load and validate declarative CST rewriting rules from YAML.

    Args:
        path (str | None): Optional path to cst_rewrite_rules.yaml.

    Returns:
        CSTRewriteRulesConfig: Validated CST rewrite configuration.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cst_rewrite_rules.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = CSTRewriteRulesRootConfig.model_validate(data)
    return root.cst_rewrite_rules
