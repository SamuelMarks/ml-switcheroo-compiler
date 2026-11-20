"""IR node and graph fixture builders for unit tests."""

from __future__ import annotations

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode


def build_test_node(
    node_id: str,
    op_type: str,
    inputs: list[str] | None = None,
    attributes: dict[str, AttributeValue] | None = None,
) -> LogicalNode:
    """Construct a typed ``LogicalNode`` fixture for unit tests.

    Args:
        node_id (str): Unique node identifier.
        op_type (str): Canonical IR operation name.
        inputs (list[str] | None): Input node IDs.
        attributes (dict[str, AttributeValue] | None): Node attribute mapping.

    Returns:
        LogicalNode: Constructed logical IR node.
    """
    return LogicalNode(id=node_id, op_type=op_type, inputs=inputs or [], attributes=attributes or {})


def build_test_graph(
    name: str = "test_graph",
    nodes: dict[str, LogicalNode] | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> LogicalGraph:
    """Construct a typed ``LogicalGraph`` fixture for unit tests.

    Args:
        name (str): Graph name.
        nodes (dict[str, LogicalNode] | None): Node dictionary keyed by ID.
        inputs (list[str] | None): Graph input IDs.
        outputs (list[str] | None): Graph output IDs.

    Returns:
        LogicalGraph: Constructed logical IR graph.
    """
    return LogicalGraph(name=name, nodes=nodes or {}, inputs=inputs or [], outputs=outputs or [])
