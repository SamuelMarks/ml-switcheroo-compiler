"""Foreign architecture bridging.

Tools for ingesting external graphs like Torch FX and JAX jaxpr.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def _handle_fx_placeholder(node: object, graph: LogicalGraph, node_map: dict[str, str]) -> None:
    node_map[node.name] = node.name


def _handle_fx_call_function(node: object, graph: LogicalGraph, node_map: dict[str, str]) -> None:
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


def _handle_fx_output(node: object, graph: LogicalGraph, node_map: dict[str, str]) -> None:
    outputs = []
    args = node.args[0]
    if isinstance(args, tuple):
        outputs.extend([arg.name for arg in args if hasattr(arg, "name")])
    elif hasattr(args, "name"):
        outputs.append(args.name)
    graph.outputs = outputs


def ingest_torch_fx(fx_graph_module: object) -> LogicalGraph:
    """Ingests a Torch FX GraphModule and converts it to a LogicalGraph.

    Args:
        fx_graph_module (object): The Torch FX GraphModule.

    Returns:
        LogicalGraph: The converted LogicalGraph.
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


def _extract_jaxpr_constants(jaxpr: object, graph: LogicalGraph) -> None:
    """Extract constants from a JAX jaxpr.

    Args:
        jaxpr (object): The JAX jaxpr.
        graph (LogicalGraph): The logical graph to populate.
    """
    pass


def _translate_jax_equation(eqn: object, graph: LogicalGraph) -> None:
    """Translate a JAX equation into a logical node.

    Args:
        eqn (object): The JAX equation.
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


def ingest_jaxpr(jaxpr: object) -> LogicalGraph:
    """Ingests a JAX jaxpr and converts it to a LogicalGraph.

    Args:
        jaxpr (object): The JAX jaxpr.

    Returns:
        LogicalGraph: The converted LogicalGraph.
    """
    if jaxpr is None:
        raise ValueError("JAX jaxpr cannot be None")

    graph = LogicalGraph(name="jaxpr_ingested")

    _extract_jaxpr_constants(jaxpr, graph)

    if hasattr(jaxpr, "eqns"):
        for eqn in jaxpr.eqns:
            _translate_jax_equation(eqn, graph)

    return graph
