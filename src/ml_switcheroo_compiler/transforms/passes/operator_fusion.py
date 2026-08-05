# ruff: noqa: C901, PLR0911, PLR0912
"""Operator fusion pass."""

from __future__ import annotations

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
    capture_map: dict[str, object],
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
    if len(node.inputs) != len(pattern.inputs):
        return False
    for i, inp_pat in enumerate(pattern.inputs):
        inp_id = node.inputs[i]
        if not match_pattern(graph, inp_id, inp_pat, capture_map):
            return False
    return True


def match_pattern(
    graph: IRGraph,
    node_id: object,
    pattern: NodePattern,
    capture_map: dict[str, object],
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

    def apply(self, graph: IRGraph, match: dict[str, object]) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The graph parameter.
            match (dict): The match parameter.
            object: Result.

        Raises:
            NotImplementedError: An exception.
        """
        raise NotImplementedError


class ReshapeReshapeFusion(FusionRule):
    """Fuses consecutive Reshape operations."""

    def __init__(self) -> None:
        """Initialize ReshapeReshapeFusion."""
        pattern = NodePattern(
            op_type="Reshape",
            capture="reshape2",
            inputs=[
                NodePattern(
                    op_type="Reshape",
                    capture="reshape1",
                    inputs=[
                        NodePattern(capture="input"),
                        NodePattern(capture="shape1"),
                    ],
                ),
                NodePattern(capture="shape2"),
            ],
        )
        super().__init__("reshape_reshape", pattern)

    def apply(self, graph: IRGraph, match: dict[str, object]) -> dict[str, IRNode] | None:
        """Apply the reshape-reshape fusion.

        Args:
            graph (IRGraph): The IRGraph.
            match (dict): The matched nodes or values.

        Returns:
            dict[str, IRNode] | None: The replacement dictionary.
        """
        reshape2 = match["reshape2"]
        input_node_or_val = match["input"]
        shape2 = match["shape2"]
        # If it's a node, get its id, otherwise use the value directly
        inp_id = input_node_or_val.id if isinstance(input_node_or_val, IRNode) else input_node_or_val
        shape2_id = shape2.id if isinstance(shape2, IRNode) else shape2
        # Bypass reshape1
        new_inputs = [inp_id, shape2_id]
        new_node = clone_logical_node(reshape2, inputs=new_inputs)
        return {reshape2.id: new_node}


class PatternMatchingEngine:
    """Engine that applies fusion rules over a graph."""

    def __init__(self, rules: list[FusionRule], cost_model: CostModel | None = None) -> None:
        """Initialize PatternMatchingEngine.

        Args:
            rules (list[FusionRule]): List of fusion rules to apply.
            cost_model (CostModel, optional): The cost model to validate fusions.
        """
        self.rules = rules
        self.cost_model = cost_model

    def _try_match_rules(self, graph: IRGraph, node_id: str, new_nodes: dict) -> bool:
        """Try applying matching rules to a single node.

        Args:
            graph (IRGraph): The graph parameter.
            node_id (str): The node_id parameter.
            new_nodes (dict): The new_nodes parameter.

        Returns:
            bool: Result.
        """
        for rule in self.rules:
            capture_map: dict[str, object] = {}
            if match_pattern(graph, node_id, rule.pattern, capture_map):
                replacements = rule.apply(graph, capture_map)
                if replacements:
                    if self.cost_model and not self.cost_model.is_fusion_valid(replacements):
                        continue
                    for rep_id, new_node in replacements.items():
                        new_nodes[rep_id] = new_node
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
        new_nodes = {}
        for node_id, node in graph.nodes.items():
            matched_rule = self._try_match_rules(graph, node_id, new_nodes)
            if matched_rule:
                optimized = True
            else:
                if node_id not in new_nodes:
                    new_nodes[node_id] = node
        if optimized:
            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return optimized


def apply_operator_fusion(graph: IRGraph) -> IRGraph:
    """Apply operator fusion pass.

    This pass fuses consecutive compatible operations using a pattern matching engine.

    Args:
        graph (IRGraph): The IR graph to optimize.

    Returns:
        IRGraph: The optimized graph.
    """
    rules = [
        ReshapeReshapeFusion(),
        ElementwiseFusion(),
        Conv2DBatchNormFusion(),
        LinearFusion(),
        MHAFusion(),
    ]
    engine = PatternMatchingEngine(rules, CostModel(max_cost=50))
    engine.apply_passes(graph)
    return graph


class ElementwiseFusion(FusionRule):
    """Fuses Add followed by Relu into AddRelu."""

    def __init__(self) -> None:
        """Initialize ElementwiseFusion."""
        pattern = NodePattern(
            op_type="Relu",
            capture="relu",
            inputs=[
                NodePattern(
                    op_type="Add",
                    capture="add",
                    inputs=[
                        NodePattern(capture="in1"),
                        NodePattern(capture="in2"),
                    ],
                )
            ],
        )
        super().__init__("add_relu", pattern)

    def apply(self, graph: IRGraph, match: dict[str, object]) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The IRGraph.
            match (dict): The matched nodes.

        Returns:
            dict: The replacement nodes.
        """
        relu = match["relu"]
        in1 = match["in1"]
        in2 = match["in2"]
        in1_id = in1.id if isinstance(in1, IRNode) else in1
        in2_id = in2.id if isinstance(in2, IRNode) else in2
        new_node = clone_logical_node(relu, inputs=[in1_id, in2_id])
        new_node.op_type = "AddRelu"
        # Only apply if Add node is not used by anything else (or we can just replace the relu, but ideally we'd remove Add if dead code, DCE will handle it)
        return {relu.id: new_node}


class Conv2DBatchNormFusion(FusionRule):
    """Fuses Conv2D followed by BatchNorm."""

    def __init__(self) -> None:
        """Initialize Conv2DBatchNormFusion."""
        pattern = NodePattern(
            op_type="BatchNorm",
            capture="bn",
            inputs=[
                NodePattern(
                    op_type="Conv2D",
                    capture="conv",
                    inputs=[
                        NodePattern(capture="in"),
                        NodePattern(capture="weight"),
                    ],
                ),
                NodePattern(capture="scale"),
                NodePattern(capture="bias"),
                NodePattern(capture="mean"),
                NodePattern(capture="var"),
            ],
        )
        super().__init__("conv2d_batchnorm", pattern)

    def apply(self, graph: IRGraph, match: dict[str, object]) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The IRGraph.
            match (dict): The matched nodes.

        Returns:
            dict: The replacement nodes.
        """
        bn = match["bn"]
        in_id = match["in"].id if isinstance(match["in"], IRNode) else match["in"]
        weight_id = match["weight"].id if isinstance(match["weight"], IRNode) else match["weight"]
        scale_id = match["scale"].id if isinstance(match["scale"], IRNode) else match["scale"]
        bias_id = match["bias"].id if isinstance(match["bias"], IRNode) else match["bias"]
        mean_id = match["mean"].id if isinstance(match["mean"], IRNode) else match["mean"]
        var_id = match["var"].id if isinstance(match["var"], IRNode) else match["var"]
        new_node = clone_logical_node(bn, inputs=[in_id, weight_id, scale_id, bias_id, mean_id, var_id])
        new_node.op_type = "Conv2DBatchNorm"
        return {bn.id: new_node}


class LinearFusion(FusionRule):
    """Fuses MatMul followed by BiasAdd into Linear."""

    def __init__(self) -> None:
        """Initialize LinearFusion."""
        pattern = NodePattern(
            op_type="Add",  # Often bias add is just an Add
            capture="add",
            inputs=[
                NodePattern(
                    op_type="MatMul",
                    capture="matmul",
                    inputs=[
                        NodePattern(capture="in1"),
                        NodePattern(capture="in2"),
                    ],
                ),
                NodePattern(capture="bias"),
            ],
        )
        super().__init__("linear", pattern)

    def apply(self, graph: IRGraph, match: dict[str, object]) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The IRGraph.
            match (dict): The matched nodes.

        Returns:
            dict: The replacement nodes.
        """
        add = match["add"]
        in1_id = match["in1"].id if isinstance(match["in1"], IRNode) else match["in1"]
        in2_id = match["in2"].id if isinstance(match["in2"], IRNode) else match["in2"]
        bias_id = match["bias"].id if isinstance(match["bias"], IRNode) else match["bias"]
        new_node = clone_logical_node(add, inputs=[in1_id, in2_id, bias_id])
        new_node.op_type = "Linear"
        return {add.id: new_node}


class MHAFusion(FusionRule):
    """Fuses Q, K, V MatMuls and Softmax into MultiHeadAttention."""

    def __init__(self) -> None:
        """Initialize MHAFusion."""
        # Simplified MHA pattern matching Softmax(Q * K^T) * V
        pattern = NodePattern(
            op_type="MatMul",
            capture="matmul2",
            inputs=[
                NodePattern(
                    op_type="Softmax",
                    capture="softmax",
                    inputs=[
                        NodePattern(
                            op_type="MatMul",
                            capture="matmul1",
                            inputs=[
                                NodePattern(capture="q"),
                                NodePattern(capture="k"),
                            ],
                        )
                    ],
                ),
                NodePattern(capture="v"),
            ],
        )
        super().__init__("mha", pattern)

    def apply(self, graph: IRGraph, match: dict[str, object]) -> dict[str, IRNode] | None:
        """Apply the fusion rule.

        Args:
            graph (IRGraph): The IRGraph.
            match (dict): The matched nodes.

        Returns:
            dict: The replacement nodes.
        """
        matmul2 = match["matmul2"]
        q_id = match["q"].id if isinstance(match["q"], IRNode) else match["q"]
        k_id = match["k"].id if isinstance(match["k"], IRNode) else match["k"]
        v_id = match["v"].id if isinstance(match["v"], IRNode) else match["v"]
        new_node = clone_logical_node(matmul2, inputs=[q_id, k_id, v_id])
        new_node.op_type = "MultiHeadAttention"
        return {matmul2.id: new_node}


def estimate_node_cost(node: IRNode) -> int:
    """Estimate a simplified cost for an IR node.

    Args:
        node (IRNode): The node to evaluate.

    Returns:
        int: The cost value.
    """
    cost_map = {
        "Reshape": 0,
        "Relu": 1,
        "Add": 1,
        "MatMul": 10,
        "Conv2D": 20,
        "BatchNorm": 5,
        "Softmax": 5,
        "Linear": 11,
        "Conv2DBatchNorm": 25,
        "MultiHeadAttention": 30,
        "AddRelu": 2,
    }
    return cost_map.get(node.op_type, 1)


class CostModel:
    """Cost model to ensure fusions don't exceed backend limits."""

    def __init__(self, max_cost: int = 50) -> None:
        """Initialize CostModel.

        Args:
            max_cost (int): Maximum allowed cost for a fused node.
        """
        self.max_cost = max_cost

    def is_fusion_valid(self, replacement_nodes: dict[str, IRNode]) -> bool:
        """Check if the fused nodes are within cost limits.

        Args:
            replacement_nodes (dict): The proposed replacement nodes.

        Returns:
            bool: True if valid.
        """
        for node in replacement_nodes.values():
            if estimate_node_cost(node) > self.max_cost:
                return False
        return True


# Ensure we update the PatternMatchingEngine to use the cost model
