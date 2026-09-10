"""Operator fusion pass for graph optimization."""

from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
import glob
import os

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node
from ml_switcheroo_compiler.transforms.passes.config_models import (
    ComputeCosts,
    CostModelConfig,
    FusionPatternConfig,
    NodePatternConfig,
    PassConfig,
)


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
        self.op_type: str | None = op_type
        self.capture: str | None = capture
        self.inputs: list[NodePattern] | None = inputs


def _match_node_inputs(
    graph: IRGraph,
    node: IRNode,
    pattern: NodePattern,
    capture_map: dict[str, str | IRNode],
) -> bool:
    """Match the inputs of a node against a pattern.

    Args:
        graph (IRGraph): The graph parameter.
        node (IRNode): The node parameter.
        pattern (NodePattern): The pattern parameter.
        capture_map (dict): The capture_map parameter.

    Returns:
        bool: True if inputs match, False otherwise.
    """
    p_inputs: list[NodePattern] = pattern.inputs or []
    if len(node.inputs) != len(p_inputs):
        return False
    for i, inp_pat in enumerate(p_inputs):
        inp_id: str = node.inputs[i]
        if not match_pattern(graph, inp_id, inp_pat, capture_map):
            return False
    return True


def match_pattern(
    graph: IRGraph,
    node_id: str | IRNode | None,
    pattern: NodePattern,
    capture_map: dict[str, str | IRNode],
) -> bool:
    """Recursively match a pattern starting from a specific node ID or value in the graph.

    Args:
        graph (IRGraph): The IRGraph.
        node_id (str | IRNode | None): The ID of the node to match, or a raw value.
        pattern (NodePattern): The pattern to match against.
        capture_map (dict): A dictionary to store captured nodes or values.

    Returns:
        bool: True if the pattern matches, False otherwise.
    """
    if not isinstance(node_id, str):
        if pattern.op_type is not None or pattern.inputs is not None:
            return False
        if pattern.capture is not None and node_id is not None:
            capture_map[pattern.capture] = node_id
        return True
    node: IRNode | None = graph.nodes.get(node_id)
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
        self.name: str = name
        self.pattern: NodePattern = pattern

    def apply(self, graph: IRGraph, match: dict[str, str | IRNode]) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The graph parameter.
            match (dict): The match parameter.

        Returns:
            dict[str, IRNode] | None: Result mapping or None.
        """
        return None


class PatternMatchingEngine:
    """Engine that applies fusion rules over a graph."""

    def __init__(
        self,
        rules: list[FusionRule],
        cost_model: MemoryAwareCostModel | None = None,
    ) -> None:
        """Initialize PatternMatchingEngine.

        Args:
            rules (list[FusionRule]): List of fusion rules to apply.
            cost_model (MemoryAwareCostModel, optional): The cost model to validate fusions.
        """
        self.rules: list[FusionRule] = rules
        self.cost_model: MemoryAwareCostModel | None = cost_model

    def _try_match_rules(
        self,
        graph: IRGraph,
        node_id: str,
        new_nodes: dict[str, IRNode],
        id_map: dict[str, str],
    ) -> bool:
        """Try applying matching rules to a single node.

        Args:
            graph (IRGraph): The graph parameter.
            node_id (str): The node_id parameter.
            new_nodes (dict): The new_nodes parameter.
            id_map (dict): The id_map parameter.

        Returns:
            bool: True if a rule matched and was applied, False otherwise.
        """
        for rule in self.rules:
            capture_map: dict[str, str | IRNode] = {}
            if match_pattern(graph, node_id, rule.pattern, capture_map):
                replacements: dict[str, IRNode] | None = rule.apply(graph, capture_map)
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
            bool: True if the graph was modified, False otherwise.
        """
        optimized: bool = False
        new_nodes: dict[str, IRNode] = {}
        id_map: dict[str, str] = {}

        for node_id, node in graph.nodes.items():
            matched_rule: bool = self._try_match_rules(graph, node_id, new_nodes, id_map)
            if matched_rule:
                optimized = True
            else:
                if node_id not in new_nodes:
                    new_nodes[node_id] = node

        if optimized:
            for n in new_nodes.values():
                for i, in_id in enumerate(n.inputs):
                    if in_id in id_map:
                        n.inputs[i] = id_map[in_id]

            if hasattr(graph, "inputs"):
                for i, in_id in enumerate(graph.inputs):
                    if in_id in id_map:
                        graph.inputs[i] = id_map[in_id]
            for i, out_id in enumerate(graph.outputs):
                if out_id in id_map:
                    graph.outputs[i] = id_map[out_id]

            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return optimized


def _load_pass_config() -> PassConfig:
    """Load pass configuration from YAML file.

    Returns:
        PassConfig: The loaded pass config object.
    """
    yaml_path: str = os.path.join(os.path.dirname(__file__), "pass_config.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            res = yaml.safe_load(f)
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


def _discover_fusion_patterns(patterns_dir: str | None = None) -> list[FusionRule]:
    """Automatically discover and load all fusion pattern YAMLs from fusion_patterns directory.

    Args:
        patterns_dir (str, optional): Directory containing fusion pattern YAML files.

    Returns:
        list[FusionRule]: Loaded fusion rules from declarative pattern files.
    """
    rules: list[FusionRule] = []
    if patterns_dir is None:
        patterns_dir = os.path.join(os.path.dirname(__file__), "fusion_patterns")
    if os.path.isdir(patterns_dir):
        yaml_files = sorted(glob.glob(os.path.join(patterns_dir, "*.yaml")))
        for yf in yaml_files:
            try:
                with open(yf) as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        for rule_name, rule_data in data.items():
                            if isinstance(rule_data, dict) and "pattern" in rule_data and "replacement" in rule_data:
                                p_config = FusionPatternConfig(**rule_data)
                                rules.append(YamlFusionRule(rule_name, p_config))
            except Exception:
                continue
    return rules


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

    def apply(self, graph: IRGraph, match: dict[str, str | IRNode]) -> dict[str, IRNode] | None:
        """Apply fusion rule.

        Args:
            graph (IRGraph): The IR graph.
            match (dict[str, str | IRNode]): Matched dictionary.

        Returns:
            dict[str, IRNode] | None: Replaced nodes or None.
        """
        replacement = self.config.replacement
        target: str | IRNode | None = match.get(replacement.capture_to_replace)
        if not isinstance(target, IRNode):
            return None

        new_inputs: list[str] = []
        for inp in replacement.inputs:
            val: str | IRNode | None = match.get(inp)
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
        config: dict[str, dict[str, float | int | dict[str, float | int]] | float | int] | None,
    ) -> None:
        """Initialize the memory-aware cost model.

        Args:
            config (dict, optional): Cost model configuration mapping.
        """
        self.config: dict[str, dict[str, float | int | dict[str, float | int]] | float | int] | None = config

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
            max_memory: int = int(self.config["max_fusion_memory_bytes"])
        elif "compute_intensity_metrics" in self.config and isinstance(self.config["compute_intensity_metrics"], dict):
            max_memory = int(self.config["compute_intensity_metrics"].get("max_fusion_memory_bytes", 1024 * 1024 * 512))
        else:
            max_memory = 1024 * 1024 * 512

        total_mem: int = 0
        memory_sizes = self.config.get("memory_sizes", {})

        for node in replacements.values():
            shape: tuple[int | str, ...] | None = getattr(node, "shape_metadata", None)
            if shape and not getattr(node, "is_dynamic_shape", False):
                if any(isinstance(d, str) for d in shape):
                    continue

                elements: int = 1
                for dim in shape:
                    elements *= max(1, int(dim))

                dtype: str = str(node.attributes.get("dtype", "float32"))
                dtype_size: int = 4
                if isinstance(memory_sizes, dict):
                    dtype_size = int(memory_sizes.get(dtype, 4))
                total_mem += elements * dtype_size

        return bool(total_mem <= max_memory)


ELEMENTWISE_OPS: set[str] = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "TrueDivide",
    "Relu",
    "Sigmoid",
    "Neg",
    "Abs",
    "Exp",
    "Log",
    "Sqrt",
    "Gelu",
    "Silu",
    "Tanh",
    "Pow",
    "Maximum",
    "Minimum",
}


def _get_scalar_expression_snippet(op_type: str, args: list[str]) -> str:
    """Generate scalar expression snippet for an elementwise op.

    Args:
        op_type (str): Operation name.
        args (list[str]): Input argument identifiers.

    Returns:
        str: C/C++ style scalar expression.
    """
    op = op_type.lower()
    a = args[0] if args else "0.0f"
    b = args[1] if len(args) > 1 else "0.0f"

    if op == "add":
        return f"({a} + {b})"
    if op == "sub":
        return f"({a} - {b})"
    if op == "mul":
        return f"({a} * {b})"
    if op in ("div", "truedivide"):
        return f"({a} / {b})"
    if op == "relu":
        return f"fmaxf(0.0f, {a})"
    if op == "sigmoid":
        return f"(1.0f / (1.0f + expf(-{a})))"
    if op == "neg":
        return f"(-{a})"
    if op == "abs":
        return f"fabsf({a})"
    if op == "exp":
        return f"expf({a})"
    if op == "log":
        return f"logf({a})"
    if op == "sqrt":
        return f"sqrtf({a})"
    if op == "tanh":
        return f"tanhf({a})"
    if op == "gelu":
        return f"(0.5f * {a} * (1.0f + tanhf(0.79788456f * ({a} + 0.044715f * {a} * {a} * {a}))))"
    if op == "silu":
        return f"({a} / (1.0f + expf(-{a})))"
    if op == "maximum":
        return f"fmaxf({a}, {b})"
    if op == "minimum":
        return f"fminf({a}, {b})"
    return a


def _fuse_single_edge(
    graph: IRGraph,
    node: IRNode,
    inp_id: str,
    inp_index: int,
    producer: IRNode,
) -> None:
    """Fuse a single producer node into a consumer node.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): Consumer node to fuse into.
        inp_id (str): Producer node id to remove.
        inp_index (int): Input slot index being replaced.
        producer (IRNode): Producer node being inlined.
    """
    fused_ops: list[str] = []
    if producer.op_type == "FusedElementwise":
        fused_ops.extend(producer.attributes.get("fused_ops", []))
        p_expr: str = str(producer.attributes.get("scalar_expr", "0.0f"))
    else:
        fused_ops.append(producer.op_type)
        p_args = [f"in_{k}" for k in range(len(producer.inputs))]
        p_expr = _get_scalar_expression_snippet(producer.op_type, p_args)

    if node.op_type == "FusedElementwise":
        fused_ops.extend(node.attributes.get("fused_ops", []))
        n_expr: str = str(node.attributes.get("scalar_expr", "0.0f"))
    else:
        fused_ops.append(node.op_type)
        n_args = [f"in_{k}" for k in range(len(node.inputs))]
        n_expr = _get_scalar_expression_snippet(node.op_type, n_args)

    new_inputs: list[str] = []
    for idx_in, inp in enumerate(node.inputs):
        if idx_in == inp_index:
            new_inputs.extend(producer.inputs)
        else:
            new_inputs.append(inp)

    unique_inputs: list[str] = []
    for inp in new_inputs:
        if inp not in unique_inputs:
            unique_inputs.append(inp)

    for k, p_in in enumerate(producer.inputs):
        p_expr = p_expr.replace(f"in_{k}", f"in{unique_inputs.index(p_in)}")

    for idx_in, inp in enumerate(node.inputs):
        if idx_in == inp_index:
            n_expr = n_expr.replace(f"in_{idx_in}", p_expr)
        else:
            n_expr = n_expr.replace(f"in_{idx_in}", f"in{unique_inputs.index(inp)}")

    fused_node = IRNode(
        id=node.id,
        op_type="FusedElementwise",
        inputs=unique_inputs,
        shape_metadata=node.shape_metadata,
    )
    fused_node.attributes = {
        "fused_ops": fused_ops,
        "scalar_expr": n_expr,
    }

    del graph.nodes[inp_id]
    graph.nodes[node.id] = fused_node


def fuse_elementwise_clusters(graph: IRGraph) -> bool:
    """Fuse elementwise DAG clusters into composite FusedElementwise IR nodes.

    Args:
        graph (IRGraph): The IR computation graph to optimize.

    Returns:
        bool: True if any elementwise DAG clusters were fused, False otherwise.
    """
    if not getattr(graph, "nodes", None):
        return False

    modified: bool = False
    changed: bool = True

    while changed:
        changed = False
        consumer_counts: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        for node in graph.nodes.values():
            for inp in getattr(node, "inputs", []):
                if inp in consumer_counts:
                    consumer_counts[inp] += 1
        for out_id in getattr(graph, "outputs", []):
            if out_id in consumer_counts:
                consumer_counts[out_id] += 1

        for node in list(graph.nodes.values()):
            if node.op_type not in ELEMENTWISE_OPS and node.op_type != "FusedElementwise":
                continue

            for i, inp_id in enumerate(list(node.inputs)):
                producer = graph.nodes.get(inp_id)
                if producer is None:
                    continue
                if producer.op_type not in ELEMENTWISE_OPS and producer.op_type != "FusedElementwise":
                    continue
                if consumer_counts.get(inp_id, 0) != 1:
                    continue
                p_shape = getattr(producer, "shape_metadata", None)
                n_shape = getattr(node, "shape_metadata", None)
                if p_shape and n_shape and p_shape != n_shape:
                    continue

                _fuse_single_edge(graph, node, inp_id, i, producer)
                changed = True
                modified = True
                break
            if changed:
                break

    return modified


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

    cost_model_config: dict[str, dict[str, float | int | dict[str, float | int]] | float | int] | None = None
    cost_yaml_path: str = os.path.join(os.path.dirname(__file__), "cost_models.yaml")
    if os.path.exists(cost_yaml_path):
        with open(cost_yaml_path) as f:
            cost_model_config = yaml.safe_load(f)

    engine: PatternMatchingEngine = PatternMatchingEngine(
        rules,
        MemoryAwareCostModel(cost_model_config) if cost_model_config else None,
    )
    pattern_modified: bool = engine.apply_passes(graph)
    cluster_modified: bool = fuse_elementwise_clusters(graph)
    if pattern_modified or cluster_modified:
        dce_pass(graph)
        return True
    return False


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
