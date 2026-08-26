"""Operator fusion pass."""

from __future__ import annotations

"""Operator fusion pass."""
# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

import typing

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


class NodePattern:
    """Provide a declarative pattern for matching a single node and its inputs in an IRGraph."""

    def __init__(
        self,
        op_type: str | None = None,
        capture: str | None = None,
        inputs: list[NodePattern] | None = None,
    ) -> None:
        """Initialize NodePattern.

        Args:
            op_type (str, optional): The expected operation type.
            capture (str, optional): The key to store the matched node in the capture map.
            inputs (list[NodePattern], optional): Patterns for the expected inputs.
        """
        self.op_type = op_type
        self.capture = capture
        self.inputs = inputs


def _match_node_inputs(
    graph: IRGraph,
    node: IRNode,
    pattern: NodePattern,
    capture_map,
) -> bool:
    """Match the inputs of a node against a pattern.

    Args:
        graph (IRGraph): The graph parameter.
        node (IRNode): The node parameter.
        pattern (NodePattern): The pattern parameter.
        capture_map (dict): The capture_map parameter.

    Returns:
        bool: Result.
    """
    p_inputs = pattern.inputs or []
    if len(node.inputs) != len(p_inputs):
        return False
    for i, inp_pat in enumerate(p_inputs):
        inp_id = node.inputs[i]
        if not match_pattern(graph, inp_id, inp_pat, capture_map):
            return False
    return True


def match_pattern(
    graph: IRGraph,
    node_id,
    pattern: NodePattern,
    capture_map,
) -> bool:
    """Recursively match a pattern starting from a specific node ID or value in the graph.

    Args:
        graph (IRGraph): The IRGraph.
        node_id (object): The ID of the node to match, or a raw value.
        pattern (NodePattern): The pattern to match against.
        capture_map (dict): A dictionary to store captured nodes or values.

    Returns:
        bool: True if the pattern matches, False otherwise.
    """
    if not isinstance(node_id, str):
        if pattern.op_type is not None or pattern.inputs is not None:
            return False
        if pattern.capture is not None:
            capture_map[pattern.capture] = node_id
        return True
    node = graph.nodes.get(node_id)
    if not node or (pattern.op_type is not None and node.op_type != pattern.op_type):
        return False
    if pattern.capture is not None:
        capture_map[pattern.capture] = node
    if pattern.inputs is not None:
        return _match_node_inputs(graph, node, pattern, capture_map)
    return True


class FusionRule:
    """Define base class for a fusion rule."""

    def __init__(self, name: str, pattern: NodePattern) -> None:
        """Initialize FusionRule.

        Args:
            name (str): The name of the fusion rule.
            pattern (NodePattern): The root pattern of the fusion rule.
        """
        self.name = name
        self.pattern = pattern

    def apply(self, graph: IRGraph, match) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The graph parameter.
            match (dict): The match parameter.
            object: Result.

        """
        return None


class PatternMatchingEngine:
    """Engine that applies fusion rules over a graph."""

    def __init__(self, rules: list[FusionRule], cost_model=None) -> None:
        """Initialize PatternMatchingEngine.

        Args:
            rules (list[FusionRule]): List of fusion rules to apply.
            cost_model (CostModel, optional): The cost model to validate fusions.
        """
        self.rules = rules
        self.cost_model = cost_model

    def _try_match_rules(self, graph: IRGraph, node_id: str, new_nodes, id_map) -> bool:
        """Try applying matching rules to a single node.

        Args:
            graph (IRGraph): The graph parameter.
            node_id (str): The node_id parameter.
            new_nodes (dict): The new_nodes parameter.
            id_map (dict): The id_map parameter.

        Returns:
            bool: Result.
        """
        for rule in self.rules:
            capture_map = {}
            if match_pattern(graph, node_id, rule.pattern, capture_map):
                replacements = rule.apply(graph, capture_map)
                if replacements:
                    if self.cost_model and not self.cost_model.is_fusion_valid(replacements):
                        continue
                    for old_id, new_node in replacements.items():
                        new_nodes[new_node.id] = new_node
                        if old_id != new_node.id:
                            id_map[old_id] = new_node.id
                    return True
        return False

    def apply_passes(self, graph: IRGraph) -> bool:
        """Apply the pattern matching rules to the graph.

        Args:
            graph (IRGraph): The IRGraph to optimize.

        Returns:
            bool: True if the graph was modified.
        """
        optimized = False
        new_nodes: dict[str, IRNode] = {}
        id_map: dict[str, str] = {}

        for node_id, node in graph.nodes.items():
            matched_rule = self._try_match_rules(graph, node_id, new_nodes, id_map)
            if matched_rule:
                optimized = True
            else:
                if node_id not in new_nodes:
                    new_nodes[node_id] = node

        if optimized:
            # Explicit Edge Rewiring
            for n in new_nodes.values():
                for i, in_id in enumerate(n.inputs):
                    if in_id in id_map:
                        n.inputs[i] = id_map[in_id]

            if hasattr(graph, "inputs"):
                for i, in_id in enumerate(graph.inputs):
                    if in_id in id_map:
                        graph.inputs[i] = id_map[in_id]
            # Also update graph outputs
            for i, out_id in enumerate(graph.outputs):
                if out_id in id_map:
                    graph.outputs[i] = id_map[out_id]

            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return optimized


def _load_pass_config():
    """Load pass configuration from YAML file.

    Returns:
        The loaded pass config dictionary.
    """
    import os

    import yaml

    from ml_switcheroo_compiler.transforms.passes.config_models import PassConfig

    yaml_path = os.path.join(os.path.dirname(__file__), "pass_config.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            res = yaml.safe_load(f)
            return PassConfig(**res)
    # Return a default empty config or fail.
    from ml_switcheroo_compiler.transforms.passes.config_models import CostModelConfig

    return PassConfig(execution_order=[], cost_model=CostModelConfig(memory_costs={}, compute_costs={}, default_memory_cost=0, default_compute_cost=0), fusion_patterns={})


from ml_switcheroo_compiler.transforms.passes.config_models import FusionPatternConfig, NodePatternConfig


class YamlFusionRule(FusionRule):
    """YAML-based fusion rule."""

    def __init__(self, name: str, config: FusionPatternConfig) -> None:
        """Initialize YAML fusion rule.

        Args:
            name (str): Rule name.
            config (FusionPatternConfig): Rule config.
        """
        self.config = config
        pattern = self._build_pattern(config.pattern)
        super().__init__(name, pattern)

    def _build_pattern(self, p: NodePatternConfig) -> NodePattern:
        """Build pattern from pydantic model.

        Args:
            p (NodePatternConfig): Pattern config.

        Returns:
            NodePattern: Built pattern.
        """
        inputs = None
        if p.inputs is not None:
            inputs = [self._build_pattern(ip) for ip in p.inputs]
        return NodePattern(op_type=p.op_type, capture=p.capture, inputs=inputs)

    def apply(self, graph: IRGraph, match) -> dict[str, IRNode] | None:
        """Apply fusion rule.

        Args:
            graph (IRGraph): The IR graph.
            match (dict[str, object]): Matched dictionary.

        Returns:
            dict[str, IRNode] | None: Replaced nodes or None.
        """
        from ml_switcheroo_compiler.ir.core import clone_logical_node

        replacement = self.config.replacement
        target = match.get(replacement.capture_to_replace)
        if not isinstance(target, IRNode):
            return None

        new_inputs = []
        for inp in replacement.inputs:
            val = match.get(inp)
            if isinstance(val, IRNode):
                new_inputs.append(val.id)
            else:
                new_inputs.append(val)

        new_node = clone_logical_node(target, inputs=new_inputs)
        new_node.op_type = replacement.op_type
        return {target.id: new_node}


class MemoryAwareCostModel:
    """A memory-aware cost model for validating operator fusion."""

    def __init__(self, config) -> None:
        """Initialize the memory-aware cost model."""
        self.config = config

    def is_fusion_valid(self, replacements) -> bool:
        """Check if fusion is valid by checking max memory thresholds."""
        if not self.config:
            return True

        max_memory = self.config.get("max_fusion_memory_bytes", 1024 * 1024 * 512)  # 512MB default
        total_mem = 0

        for node in replacements.values():
            shape = getattr(node, "shape_metadata", None)
            if shape and not getattr(node, "is_dynamic_shape", False):
                # Check for symbolic dimensions
                if any(isinstance(d, str) for d in shape):
                    continue  # Skip memory check for symbolic shapes

                elements = 1
                for dim in shape:
                    elements *= max(1, int(dim))

                dtype = node.attributes.get("dtype", "float32")
                dtype_size = int(self.config.get("memory_sizes", {}).get(dtype, 4))
                total_mem += elements * dtype_size

        return bool(total_mem <= max_memory)


def apply_operator_fusion(graph: IRGraph) -> IRGraph:
    """Apply operator fusion pass.

    This pass fuses consecutive compatible operations using a pattern matching engine.

    Args:
        graph (IRGraph): The IR graph to optimize.

    Returns:
        IRGraph: The optimized graph.
    """
    from ml_switcheroo_compiler.transforms.passes.dce import dce_pass

    rules: list[FusionRule] = []
    # Load YAML rules
    config = _load_pass_config()
    if config.fusion_patterns:
        for name, rule_config in config.fusion_patterns.items():
            rules.append(YamlFusionRule(name, rule_config))

    # Load cost models
    import os

    import yaml

    cost_model_config = None
    cost_yaml_path = os.path.join(os.path.dirname(__file__), "cost_models.yaml")
    if os.path.exists(cost_yaml_path):
        with open(cost_yaml_path) as f:
            cost_model_config = yaml.safe_load(f)

    engine = PatternMatchingEngine(rules, MemoryAwareCostModel(cost_model_config) if cost_model_config else None)
    if engine.apply_passes(graph):
        dce_pass(graph)
    return graph
