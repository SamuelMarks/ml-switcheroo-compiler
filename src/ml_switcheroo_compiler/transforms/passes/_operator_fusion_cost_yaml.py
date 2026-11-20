"""YAML fusion rule discovery and memory-aware roofline cost model."""

from __future__ import annotations

import glob
import logging
import os
import typing

import yaml
from ml_switcheroo_ir import IRGraph, IRNode

import ml_switcheroo_compiler.transforms.passes.operator_fusion as _of_mod
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ir.core import clone_logical_node
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    FusionRule,
    MatchMapType,
    NodePattern,
    _DynamicShapeNodeProtocol,
    _to_fusion_int,
)
from ml_switcheroo_compiler.transforms.passes.config_models import (
    ComputeCosts,
    CostModelConfig,
    FusionPatternConfig,
    NodePatternConfig,
    PassConfig,
)


def _load_pass_config() -> PassConfig:
    """Load pass configuration from YAML file.

    Returns:
        PassConfig: The loaded pass config object.
    """
    yaml_path: str = os.path.join(os.path.dirname(__file__), "pass_config.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            res = _of_mod.yaml.safe_load(f)
        if isinstance(res, dict):
            return PassConfig(**res)
    return PassConfig(
        execution_order=[],
        cost_model=CostModelConfig(
            memory_sizes={},
            compute_costs=ComputeCosts(
                heavy_ops=[],
                light_ops=[],
                heavy_cost=1000,
                light_cost=10,
                default_cost=50,
            ),
            compute_heavy_threshold=100,
            heavy_interleave_penalty=500,
            light_interleave_penalty=100,
        ),
        fusion_patterns={},
    )


def _load_yaml_fusion_dir(patterns_dir: str) -> list[FusionRule]:
    """Load declarative YAML fusion rules from a directory.

    Args:
        patterns_dir (str): Directory containing fusion pattern YAML files.

    Returns:
        list[FusionRule]: List of parsed YamlFusionRule instances.
    """
    rules: list[FusionRule] = []
    if not os.path.isdir(patterns_dir):
        return rules

    for yf in sorted(glob.glob(os.path.join(patterns_dir, "*.yaml"))):
        try:
            with open(yf) as f:
                data = _of_mod.yaml.safe_load(f)
            if isinstance(data, dict):
                for rule_name, rule_data in data.items():
                    if isinstance(rule_data, dict) and "pattern" in rule_data and "replacement" in rule_data:
                        rules.append(YamlFusionRule(rule_name, FusionPatternConfig(**rule_data)))
        except (OSError, yaml.YAMLError, ValueError, TypeError, KeyError) as exc:
            logging.getLogger(__name__).debug("Skipping invalid fusion pattern file %s: %s", yf, exc)
            continue
    return rules


def _load_global_fusion_patterns(global_yaml: str) -> list[FusionRule]:
    """Load global declarative target fusion rules from fusion_patterns.yaml.

    Args:
        global_yaml (str): Path to the global fusion_patterns.yaml file.

    Returns:
        list[FusionRule]: List of parsed DeclarativeTargetFusionRule instances.
    """
    rules: list[FusionRule] = []
    if not os.path.exists(global_yaml):
        return rules

    try:
        with open(global_yaml) as f:
            g_data = _of_mod.yaml.safe_load(f)
        if isinstance(g_data, dict) and isinstance(g_data.get("patterns"), list):
            for pat in g_data["patterns"]:
                targets = pat.get("target", [])
                if targets:
                    curr_pat = NodePattern(
                        op_type=targets[-1],
                        capture="root",
                        commutative=bool(pat.get("commutative", False)),
                        wildcard=bool(pat.get("wildcard", False)),
                        broadcast_dims=bool(pat.get("broadcast_dims", False)),
                    )
                    rules.append(
                        DeclarativeTargetFusionRule(
                            str(pat.get("name", "pattern")),
                            curr_pat,
                            str(pat.get("fused_op", "FusedOp")),
                        )
                    )
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return rules


def _discover_fusion_patterns(patterns_dir: str | None = None) -> list[FusionRule]:
    """Automatically discover and load all fusion pattern YAMLs from fusion_patterns directory.

    Args:
        patterns_dir (str | None): Directory containing fusion pattern YAML files.

    Returns:
        list[FusionRule]: Loaded fusion rules from declarative pattern files.
    """
    load_defaults = patterns_dir is None
    resolved_dir = patterns_dir or os.path.join(os.path.dirname(__file__), "fusion_patterns")
    rules = _load_yaml_fusion_dir(resolved_dir)

    if load_defaults:
        global_yaml = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fusion_patterns.yaml")
        rules.extend(_load_global_fusion_patterns(global_yaml))

    return rules


class DeclarativeTargetFusionRule(FusionRule):
    """Fusion rule constructed from target sequence in fusion_patterns.yaml."""

    def __init__(self, name: str, pattern: NodePattern, fused_op: str) -> None:
        """Initialize DeclarativeTargetFusionRule.

        Args:
            name (str): Rule name.
            pattern (NodePattern): Pattern to match.
            fused_op (str): Fused operation type name to emit.
        """
        super().__init__(name, pattern)
        self.fused_op: str = fused_op

    def apply(self, graph: IRGraph, match: MatchMapType) -> dict[str, IRNode] | None:
        """Apply fusion rule emitting fused operator node.

        Args:
            graph (IRGraph): IR graph to mutate.
            match (MatchMapType): Matched nodes mapping.

        Returns:
            dict[str, IRNode] | None: Replacement mapping or None.
        """
        root_node = match.get("root")
        if not isinstance(root_node, IRNode):
            return None
        new_id = f"{root_node.id}_fused"
        fused_node = clone_logical_node(root_node)
        fused_node.id = new_id
        fused_node.op_type = self.fused_op
        fused_node.attributes["fused_pattern"] = self.name
        return {root_node.id: fused_node}


class YamlFusionRule(FusionRule):
    """YAML-based fusion rule."""

    def __init__(self, name: str, config: FusionPatternConfig) -> None:
        """Initialize YAML fusion rule.

        Args:
            name (str): Rule name.
            config (FusionPatternConfig): Rule config.
        """
        self.config: FusionPatternConfig = config
        pattern: NodePattern = self._build_pattern(config.pattern)
        super().__init__(name, pattern)

    def _build_pattern(self, p: NodePatternConfig) -> NodePattern:
        """Build pattern from pydantic model.

        Args:
            p (NodePatternConfig): Pattern config.

        Returns:
            NodePattern: Built pattern.
        """
        inputs: list[NodePattern] | None = None
        if p.inputs is not None:
            inputs = [self._build_pattern(ip) for ip in p.inputs]
        return NodePattern(op_type=p.op_type, capture=p.capture, inputs=inputs)

    def apply(self, graph: IRGraph, match: MatchMapType) -> dict[str, IRNode] | None:
        """Apply fusion rule.

        Args:
            graph (IRGraph): The IR graph.
            match (MatchMapType): Matched dictionary.

        Returns:
            dict[str, IRNode] | None: Replaced nodes or None.
        """
        replacement = self.config.replacement
        target: str | IRNode | int | float | None = match.get(replacement.capture_to_replace)
        if not isinstance(target, IRNode):
            return None

        new_inputs: list[str] = []
        for inp in replacement.inputs:
            val: str | IRNode | int | float | None = match.get(inp)
            if isinstance(val, IRNode):
                new_inputs.append(val.id)
            elif isinstance(val, str):
                new_inputs.append(val)

        new_node: IRNode = clone_logical_node(target, inputs=new_inputs)
        new_node.op_type = replacement.op_type
        return {target.id: new_node}


class MemoryAwareCostModel:
    """A memory-aware cost model for validating operator fusion."""

    def __init__(
        self,
        config: typing.Mapping[str, OpValue] | None,
    ) -> None:
        """Initialize the memory-aware cost model.

        Args:
            config (typing.Mapping[str, OpValue] | None):
                Cost model configuration mapping.
        """
        self.config: typing.Mapping[str, OpValue] | None = config

    def is_fusion_valid(self, replacements: dict[str, IRNode]) -> bool:
        """Check if fusion is valid by checking max memory thresholds.

        Args:
            replacements (dict[str, IRNode]): Mapping of candidate replacement nodes.

        Returns:
            bool: True if fusion is permitted by memory budget, False otherwise.
        """
        if not self.config:
            return True

        if "max_fusion_memory_bytes" in self.config:
            max_memory: int = _to_fusion_int(self.config["max_fusion_memory_bytes"], 1024 * 1024 * 512)
        elif "compute_intensity_metrics" in self.config and isinstance(self.config["compute_intensity_metrics"], dict):
            max_memory = _to_fusion_int(
                self.config["compute_intensity_metrics"].get("max_fusion_memory_bytes", 1024 * 1024 * 512),
                1024 * 1024 * 512,
            )
        else:
            max_memory = 1024 * 1024 * 512

        total_mem: int = 0
        memory_sizes = self.config.get("memory_sizes", {})

        for node in replacements.values():
            shape = node.shape_metadata
            is_dyn = bool(node.is_dynamic_shape) if isinstance(node, _DynamicShapeNodeProtocol) else False
            if shape and not is_dyn:
                if any(isinstance(d, str) for d in shape):
                    continue

                elements: int = 1
                for dim in shape:
                    elements *= max(1, _to_fusion_int(dim, 1))

                dtype: str = str(node.attributes.get("dtype", "float32"))
                dtype_size: int = 4
                if isinstance(memory_sizes, dict):
                    dtype_size = _to_fusion_int(memory_sizes.get(dtype, 4), 4)
                total_mem += elements * dtype_size

        return bool(total_mem <= max_memory)
