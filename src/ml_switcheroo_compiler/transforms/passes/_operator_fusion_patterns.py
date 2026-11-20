"""Pattern matching primitives and fusion rule base classes for operator fusion."""

from __future__ import annotations

import typing
from collections.abc import Mapping, MutableMapping, Sequence

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ir.core import clone_logical_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.transforms.passes._operator_fusion_cost_yaml import MemoryAwareCostModel
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig

MatchMapType = dict[str, typing.Union[str, IRNode, int, float, None]]  # noqa: UP007


def _is_fusion_num(
    val: OpValue,
) -> typing.TypeGuard[int | float]:
    """Type guard narrowing a configuration or shape value to `int | float`.

    Args:
        val: Candidate numeric value.

    Returns:
        bool: True if val is an int or float.
    """
    return isinstance(val, (int, float))


def _to_fusion_int(
    val: OpValue,
    default: int = 0,
) -> int:
    """Coerce a configuration or shape value to int.

    Args:
        val: Raw numeric value.
        default (int): Fallback value if not convertible.

    Returns:
        int: Coerced integer value.
    """
    _ = default
    assert _is_fusion_num(val)
    return int(val)


@typing.runtime_checkable
class _GraphInputsProtocol(typing.Protocol):
    """Protocol for graphs exposing an `inputs` list."""

    inputs: list[str]


@typing.runtime_checkable
class _GraphOutputsProtocol(typing.Protocol):
    """Protocol for graphs exposing an `outputs` list."""

    outputs: list[str]


@typing.runtime_checkable
class _GraphNodesProtocol(typing.Protocol):
    """Protocol for graphs exposing a `nodes` mapping."""

    nodes: Mapping[str, IRNode]


@typing.runtime_checkable
class _DynamicShapeNodeProtocol(typing.Protocol):
    """Protocol for nodes exposing an `is_dynamic_shape` boolean attribute."""

    is_dynamic_shape: bool


class NodePattern:
    """Provide a declarative pattern for matching a single node and its inputs in an IRGraph."""

    def __init__(
        self,
        op_type: str | None = None,
        capture: str | None = None,
        inputs: list[NodePattern] | None = None,
        wildcard: bool = False,
        commutative: bool = False,
        broadcast_dims: bool = False,
    ) -> None:
        """Initialize NodePattern.

        Args:
            op_type (str | None): The expected operation type.
            capture (str | None): The key to store the matched node in the capture map.
            inputs (list[NodePattern] | None): Patterns for the expected inputs.
            wildcard (bool): Allow matching any operation type.
            commutative (bool): Match inputs in any permutation.
            broadcast_dims (bool): Match tensors with broadcast compatible dimensions.
        """
        self.op_type: str | None = op_type
        self.capture: str | None = capture
        self.inputs: list[NodePattern] | None = inputs
        self.wildcard: bool = wildcard or op_type in ("*", "any")
        self.commutative: bool = commutative
        self.broadcast_dims: bool = broadcast_dims


CaptureMapArg: typing.TypeAlias = typing.Union[MutableMapping[str, OpValue], MatchMapType]  # noqa: UP007


def _match_node_inputs(
    graph: IRGraph,
    node: IRNode,
    pattern: NodePattern | NodePatternConfig,
    capture_map: CaptureMapArg,
) -> bool:
    """Match the inputs of a node against a pattern supporting commutativity.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        pattern (NodePattern | NodePatternConfig): Input `pattern` argument (pattern).
        capture_map (CaptureMapArg): Input `capture_map` argument (capture map).

    Returns:
        bool: True if inputs match, False otherwise.
    """
    p_inputs: Sequence[NodePattern | NodePatternConfig] = pattern.inputs or []
    if len(node.inputs) != len(p_inputs):
        return False

    if pattern.commutative and len(p_inputs) == 2:
        orig_map = dict(capture_map)
        if match_pattern(graph, node.inputs[0], p_inputs[0], capture_map) and match_pattern(
            graph, node.inputs[1], p_inputs[1], capture_map
        ):
            return True
        capture_map.clear()
        for k_orig, v_orig in orig_map.items():
            if isinstance(v_orig, (str, int, float, IRNode)):
                _store_capture(capture_map, k_orig, v_orig)
        if match_pattern(graph, node.inputs[1], p_inputs[0], capture_map) and match_pattern(
            graph, node.inputs[0], p_inputs[1], capture_map
        ):
            return True
        return False

    for i, inp_pat in enumerate(p_inputs):
        inp_id: str = node.inputs[i]
        if not match_pattern(graph, inp_id, inp_pat, capture_map):
            return False
    return True


def match_pattern(
    graph: IRGraph | None,
    node_id: str | int | float | IRNode | None,
    pattern: NodePattern | NodePatternConfig,
    capture_map: CaptureMapArg,
) -> bool:
    """Recursively match a pattern starting from a specific node ID or value in the graph.

    Args:
        graph (IRGraph | None): The IRGraph.
        node_id (str | int | float | IRNode | None): The ID of the node to match, or a raw value.
        pattern (NodePattern | NodePatternConfig): The pattern to match against.
        capture_map (CaptureMapArg): A dictionary to store captured nodes or values.

    Returns:
        bool: True if the pattern matches, False otherwise.
    """
    is_wildcard = pattern.wildcard or pattern.op_type in ("*", "any")
    if not isinstance(node_id, str):
        if (pattern.op_type is not None and not is_wildcard) or pattern.inputs is not None:
            return False
        if pattern.capture is not None and node_id is not None:
            _store_capture(capture_map, pattern.capture, node_id)
        return True
    node: IRNode | None = graph.nodes.get(node_id) if graph is not None else None
    if node is None or (not is_wildcard and pattern.op_type is not None and node.op_type != pattern.op_type):
        return False
    assert graph is not None
    if pattern.capture is not None:
        _store_capture(capture_map, pattern.capture, node)
    return _match_node_inputs(graph, node, pattern, capture_map) if pattern.inputs is not None else True


def _store_capture(
    capture_map: CaptureMapArg,
    key: str,
    val: str | int | float | IRNode,
) -> None:
    """Store a matched value into `capture_map` preserving map variance.

    Args:
        capture_map (CaptureMapArg): Capture dictionary.
        key (str): Capture variable name.
        val (str | int | float | IRNode): Captured node or scalar/string ID.
    """
    capture_map[key] = val


class FusionRule:
    """Define base class for a fusion rule."""

    def __init__(self, name: str, pattern: NodePattern | NodePatternConfig) -> None:
        """Initialize FusionRule.

        Args:
            name (str): The name of the fusion rule.
            pattern (NodePattern | NodePatternConfig): The root pattern of the fusion rule.
        """
        self.name: str = name
        self.pattern: NodePattern | NodePatternConfig = pattern

    def apply(self, graph: IRGraph, match: MatchMapType) -> dict[str, IRNode] | None:
        """Apply the fusion rule using standard graph transformation.

        Args:
            graph (IRGraph): The target IR computation graph.
            match (MatchMapType): Mapping of captured pattern names to IR nodes or node IDs.

        Returns:
            dict[str, IRNode] | None: Mapping of replaced node IDs to fused replacement nodes, or None if match is
                empty or invalid.
        """
        if not match:
            return None
        root_node: IRNode | None = None
        target = match.get("root")
        if isinstance(target, IRNode):
            root_node = target
        else:
            for val in match.values():
                if isinstance(val, IRNode):
                    root_node = val
                    break
        if not isinstance(root_node, IRNode):
            return None

        new_id = f"{root_node.id}_fused"
        fused_node = clone_logical_node(root_node)
        fused_node.id = new_id
        fused_node.op_type = f"Fused_{self.name}" if not self.name.startswith("Fused_") else self.name
        fused_node.attributes["fused_pattern"] = self.name
        return {root_node.id: fused_node}


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
            cost_model (MemoryAwareCostModel | None): The cost model to validate fusions.
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
            graph (IRGraph): Input `graph` argument (graph).
            node_id (str): Input `node_id` argument (node id).
            new_nodes (dict[str, IRNode]): Input `new_nodes` argument (new nodes).
            id_map (dict[str, str]): Input `id_map` argument (id map).

        Returns:
            bool: True if a rule matched and was applied, False otherwise.
        """
        for rule in self.rules:
            capture_map: MatchMapType = {}
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

    @staticmethod
    def _rewire_replaced_ids(
        graph: IRGraph,
        new_nodes: dict[str, IRNode],
        id_map: dict[str, str],
    ) -> None:
        """Update node inputs and graph I/O references using the replaced node ID map.

        Args:
            graph (IRGraph): The target IRGraph being mutated.
            new_nodes (dict[str, IRNode]): Updated dictionary of IRNodes.
            id_map (dict[str, str]): Mapping from old node IDs to fused node IDs.
        """
        for n in new_nodes.values():
            n.inputs = [id_map.get(in_id, in_id) for in_id in n.inputs]

        raw_graph: _GraphInputsProtocol | str = graph
        if isinstance(raw_graph, _GraphInputsProtocol) and raw_graph.inputs is not None:
            raw_graph.inputs = [id_map.get(in_id, in_id) for in_id in raw_graph.inputs]
        graph.outputs = [id_map.get(out_id, out_id) for out_id in graph.outputs]

        graph.nodes.clear()
        graph.nodes.update(new_nodes)

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
            if self._try_match_rules(graph, node_id, new_nodes, id_map):
                optimized = True
            elif node_id not in new_nodes:
                new_nodes[node_id] = node

        if optimized:
            self._rewire_replaced_ids(graph, new_nodes, id_map)
        return optimized
