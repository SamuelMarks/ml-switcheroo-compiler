"""Operator fusion pass for graph optimization."""

from __future__ import annotations

import os

import yaml as yaml
from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _SCALAR_EXPR_TEMPLATES as _SCALAR_EXPR_TEMPLATES,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    ELEMENTWISE_OPS as ELEMENTWISE_OPS,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _count_consumers as _count_consumers,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _extract_node_ops_and_expr as _extract_node_ops_and_expr,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _fuse_single_edge as _fuse_single_edge,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _get_scalar_expression_snippet as _get_scalar_expression_snippet,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _has_graph_nodes as _has_graph_nodes,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _is_fusable_elementwise_node as _is_fusable_elementwise_node,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    _try_fuse_cluster_step as _try_fuse_cluster_step,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    fuse_elementwise_clusters as fuse_elementwise_clusters,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    DeclarativeTargetFusionRule as DeclarativeTargetFusionRule,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    MemoryAwareCostModel as MemoryAwareCostModel,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    YamlFusionRule as YamlFusionRule,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    _discover_fusion_patterns as _discover_fusion_patterns,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    _load_global_fusion_patterns as _load_global_fusion_patterns,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    _load_pass_config as _load_pass_config,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import (
    _load_yaml_fusion_dir as _load_yaml_fusion_dir,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    HorizontalFusionPass as HorizontalFusionPass,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    VerticalFusionPass as VerticalFusionPass,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    _find_conv_and_bias_inputs as _find_conv_and_bias_inputs,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    _try_fuse_conv_bias_act_node as _try_fuse_conv_bias_act_node,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    _try_fuse_elem_act_node as _try_fuse_elem_act_node,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    fuse_conv_bias_relu as fuse_conv_bias_relu,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    fuse_elementwise_activation as fuse_elementwise_activation,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_passes import (
    fuse_horizontal_patterns as fuse_horizontal_patterns,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    FusionRule as FusionRule,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    MatchMapType as MatchMapType,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    NodePattern as NodePattern,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    PatternMatchingEngine as PatternMatchingEngine,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _DynamicShapeNodeProtocol as _DynamicShapeNodeProtocol,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _GraphInputsProtocol as _GraphInputsProtocol,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _GraphNodesProtocol as _GraphNodesProtocol,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _GraphOutputsProtocol as _GraphOutputsProtocol,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _is_fusion_num as _is_fusion_num,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _match_node_inputs as _match_node_inputs,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _to_fusion_int as _to_fusion_int,
)
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    match_pattern as match_pattern,
)
from ml_switcheroo_compiler.transforms.passes.config_models import (
    PassConfig,
)


def operator_fusion_pass(graph: IRGraph) -> bool:
    """In-place operator fusion pass returning True if graph was modified.

    Args:
        graph (IRGraph): The IR graph to optimize.

    Returns:
        bool: True if fusions occurred and graph was modified, False otherwise.
    """
    from ml_switcheroo_compiler.transforms.passes.dce import dce_pass

    rules: list[FusionRule] = _discover_fusion_patterns()
    config: PassConfig = _load_pass_config()
    if config.fusion_patterns:
        for name, rule_config in config.fusion_patterns.items():
            rules.append(YamlFusionRule(name, rule_config))

    cost_model_config: dict[str, dict[str, float | int | dict[str, float | int]] | float | int | None] | None = None
    cost_yaml_path: str = os.path.join(os.path.dirname(__file__), "cost_models.yaml")
    if os.path.exists(cost_yaml_path):
        with open(cost_yaml_path) as f:
            cost_model_config = yaml.safe_load(f)

    engine: PatternMatchingEngine = PatternMatchingEngine(
        rules,
        MemoryAwareCostModel(cost_model_config) if cost_model_config else None,
    )
    pattern_modified: bool = engine.apply_passes(graph)
    if pattern_modified:
        dce_pass(graph)
    vertical_modified: bool = VerticalFusionPass().run(graph)
    if vertical_modified:
        dce_pass(graph)
    horizontal_modified: bool = HorizontalFusionPass().run(graph)
    if horizontal_modified:
        dce_pass(graph)
    cluster_modified: bool = fuse_elementwise_clusters(graph)
    if cluster_modified:
        dce_pass(graph)
    return pattern_modified or vertical_modified or horizontal_modified or cluster_modified


def apply_operator_fusion(graph: IRGraph) -> IRGraph:
    """Apply operator fusion pass.

    This pass fuses consecutive compatible operations using a pattern matching engine.

    Args:
        graph (IRGraph): The IR graph to optimize.

    Returns:
        IRGraph: The optimized graph.
    """
    operator_fusion_pass(graph)
    return graph


class OperatorFusionPass:
    """Operator fusion pass executing pattern-based graph transformations."""

    def __init__(self, patterns_dir: str | None = None) -> None:
        """Initialize OperatorFusionPass.

        Args:
            patterns_dir (str | None): Optional directory with YAML fusion patterns.
        """
        self.rules: list[FusionRule] = _discover_fusion_patterns(patterns_dir)

    def run(self, graph: IRGraph) -> bool:
        """Execute operator fusion transformations on graph.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            bool: True if graph was modified, False otherwise.
        """
        return operator_fusion_pass(graph)
