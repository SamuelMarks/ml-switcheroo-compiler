# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Foreign architecture bridging.

Tools for ingesting external graphs like Torch FX and JAX jaxpr.
"""

from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def _handle_fx_placeholder(node: Any, graph: LogicalGraph, node_map: dict[str, str]) -> None:
    """Handle a Torch FX placeholder node.

    Args:
        node (Any): The FX node.
        graph (LogicalGraph): The logical graph.
        node_map (dict[str, str]): Mapping from FX node names to graph node IDs.
    """
    node_map[node.name] = node.name


def _handle_fx_call_function(node: Any, graph: LogicalGraph, node_map: dict[str, str]) -> None:
    """Handle a Torch FX call_function node.

    Args:
        node (Any): The FX node.
        graph (LogicalGraph): The logical graph.
        node_map (dict[str, str]): Mapping from FX node names to graph node IDs.
    """
    target_name = getattr(node.target, "__name__", str(node.target))

    op_map = {
        "add": "Add",
        "mul": "Mul",
    }
    op_type = "Unknown"
    for key, val in op_map.items():
        if key in target_name:
            op_type = val
            break

    inputs = []
    for arg in node.args:
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    l_node = LogicalNode(
        id=node.name,
        op_type=op_type,
        domain="ai.onnx",
        version=1,
        attributes={},
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node.name] = l_node
    node_map[node.name] = node.name


def _handle_fx_output(node: Any, graph: LogicalGraph, node_map: dict[str, str]) -> None:
    """Handle a Torch FX output node.

    Args:
        node (Any): The FX node.
        graph (LogicalGraph): The logical graph.
        node_map (dict[str, str]): Mapping from FX node names to graph node IDs.
    """
    outputs = []
    args = node.args[0]
    if isinstance(args, tuple):
        outputs.extend([arg.name for arg in args if hasattr(arg, "name")])
    elif hasattr(args, "name"):
        outputs.append(args.name)
    graph.outputs = outputs


def ingest_torch_fx(fx_graph_module: Any) -> LogicalGraph:
    """Ingests a Torch FX GraphModule and converts it to a LogicalGraph.

    Args:
        fx_graph_module (Any): The Torch FX GraphModule.

    Returns:
        LogicalGraph: The converted LogicalGraph.

    Raises:
        ValueError: If fx_graph_module is None.
    """
    if fx_graph_module is None:
        raise ValueError("Torch FX GraphModule cannot be None")

    graph = LogicalGraph(name="torch_fx_ingested")

    if not hasattr(fx_graph_module, "graph") or not hasattr(fx_graph_module.graph, "nodes"):
        return graph

    handlers = {
        "placeholder": _handle_fx_placeholder,
        "call_function": _handle_fx_call_function,
        "output": _handle_fx_output,
    }

    node_map: dict[str, str] = {}
    for node in fx_graph_module.graph.nodes:
        if node.op in handlers:
            handlers[node.op](node, graph, node_map)

    return graph


def _extract_jaxpr_constants(jaxpr: Any, graph: LogicalGraph) -> None:
    """Extract constants from a JAX jaxpr.

    Args:
        jaxpr (Any): The JAX jaxpr.
        graph (LogicalGraph): The logical graph to populate.
    """
    if hasattr(jaxpr, "consts"):
        constvars = getattr(jaxpr, "constvars", None)
        for i, const_val in enumerate(jaxpr.consts):
            const_id = str(id(constvars[i])) if constvars else f"const_{i}"
            l_node = LogicalNode(
                id=const_id,
                op_type="Constant",
                domain="ai.onnx",
                version=1,
                attributes={"value": const_val},
                inputs=[],
                shape_metadata=getattr(const_val, "shape", ()),
            )
            graph.nodes[const_id] = l_node


def _translate_jax_equation(eqn: Any, graph: LogicalGraph) -> None:
    """Translate a JAX equation into a logical node.

    Args:
        eqn (Any): The JAX equation.
        graph (LogicalGraph): The logical graph.
    """
    op_type = "Unknown"
    primitive_name = getattr(eqn.primitive, "name", str(eqn.primitive))
    if primitive_name == "add":
        op_type = "Add"
    elif primitive_name == "mul":
        op_type = "Mul"

    inputs = []
    for invar in eqn.invars:
        inputs.append(str(id(invar)))

    out_id = str(id(eqn.outvars[0])) if eqn.outvars else "out"

    l_node = LogicalNode(
        id=out_id,
        op_type=op_type,
        domain="ai.onnx",
        version=1,
        attributes={},
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[out_id] = l_node


def ingest_jaxpr(jaxpr: Any) -> LogicalGraph:
    """Ingests a JAX jaxpr and converts it to a LogicalGraph.

    Args:
        jaxpr (Any): The JAX jaxpr.

    Returns:
        LogicalGraph: The converted LogicalGraph.

    Raises:
        ValueError: If fx_graph_module is None.
    """
    if jaxpr is None:
        raise ValueError("JAX jaxpr cannot be None")

    graph = LogicalGraph(name="jaxpr_ingested")

    _extract_jaxpr_constants(jaxpr, graph)

    if hasattr(jaxpr, "eqns"):
        for eqn in jaxpr.eqns:
            _translate_jax_equation(eqn, graph)

    return graph
