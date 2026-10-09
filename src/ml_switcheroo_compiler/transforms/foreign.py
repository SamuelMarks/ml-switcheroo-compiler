# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Foreign architecture bridging.

Tools for ingesting external graphs like Torch FX and JAX jaxpr.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def _handle_fx_placeholder(node, graph: LogicalGraph, node_map: dict[str, str], modules: dict = None) -> None:
    node_map[node.name] = node.name


def _handle_fx_call_method(node, graph: LogicalGraph, node_map: dict[str, str], modules: dict = None) -> None:
    target_name = str(node.target)

    op_map = {
        "view": "Reshape",
        "reshape": "Reshape",
        "permute": "Transpose",
        "transpose": "Transpose",
        "contiguous": "Identity",
        "relu": "Relu",
        "flatten": "Flatten",
        "unsqueeze": "Unsqueeze",
        "squeeze": "Squeeze",
    }

    op_type = op_map.get(target_name, "Unknown")

    inputs = []
    for arg in node.args:
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    attributes = {}
    if hasattr(node, "kwargs"):
        for k, v in node.kwargs.items():
            if isinstance(v, (list, tuple)):
                attributes[k] = list(v)
            elif isinstance(v, (int, float, str, bool)):
                attributes[k] = v

    l_node = LogicalNode(
        id=node.name,
        op_type=op_type,
        domain="ai.onnx",
        version=1,
        attributes=attributes,
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node.name] = l_node
    node_map[node.name] = node.name


def _handle_fx_call_module(node, graph: LogicalGraph, node_map: dict[str, str], modules: dict = None) -> None:
    target_name = str(node.target)
    op_type = "CallModule"

    if target_name in modules:
        mod = modules[target_name]
        mod_type = type(mod).__name__
        if "Conv" in mod_type:
            op_type = "Conv"
        elif "BatchNorm" in mod_type:
            op_type = "BatchNormalization"
        elif "LayerNorm" in mod_type:
            op_type = "LayerNormalization"
        elif "GroupNorm" in mod_type:
            op_type = "GroupNormalization"
        elif "Linear" in mod_type:
            op_type = "Gemm"
        elif "Dropout" in mod_type:
            op_type = "Dropout"
        elif "ReLU" in mod_type:
            op_type = "Relu"
        elif "MaxPool" in mod_type:
            op_type = "MaxPool"
        elif "AvgPool" in mod_type:
            op_type = "AveragePool"

    inputs = []
    for arg in node.args:
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    attributes = {}
    if hasattr(node, "kwargs"):
        for k, v in node.kwargs.items():
            if isinstance(v, (list, tuple)):
                attributes[k] = list(v)
            elif isinstance(v, (int, float, str, bool)):
                attributes[k] = v

    l_node = LogicalNode(
        id=node.name,
        op_type=op_type,
        domain="ai.onnx",
        version=1,
        attributes=attributes,
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node.name] = l_node
    node_map[node.name] = node.name


def _handle_fx_get_attr(node, graph: LogicalGraph, node_map: dict[str, str], modules: dict = None) -> None:
    l_node = LogicalNode(
        id=node.name,
        op_type="Constant",
        domain="ai.onnx",
        version=1,
        attributes={"target": str(node.target)},
        inputs=[],
        shape_metadata=(),
    )
    graph.nodes[node.name] = l_node
    node_map[node.name] = node.name


def _handle_fx_call_function(node, graph: LogicalGraph, node_map: dict[str, str], modules: dict = None) -> None:
    target_name = getattr(node.target, "__name__", str(node.target))

    op_map = {
        "add": "Add",
        "sub": "Sub",
        "mul": "Mul",
        "div": "Div",
        "neg": "Neg",
        "pow": "Pow",
        "abs": "Abs",
        "exp": "Exp",
        "log": "Log",
        "sqrt": "Sqrt",
        "sin": "Sin",
        "cos": "Cos",
        "tan": "Tan",
        "tanh": "Tanh",
        "sigmoid": "Sigmoid",
        "matmul": "MatMul",
        "bmm": "MatMul",
        "linear": "Gemm",
        "mm": "MatMul",
        "conv1d": "Conv",
        "conv2d": "Conv",
        "conv3d": "Conv",
        "max_pool2d": "MaxPool",
        "avg_pool2d": "AveragePool",
        "batch_norm": "BatchNormalization",
        "layer_norm": "LayerNormalization",
        "group_norm": "GroupNormalization",
        "sum": "ReduceSum",
        "mean": "ReduceMean",
        "max": "ReduceMax",
        "min": "ReduceMin",
        "argmax": "ArgMax",
        "argmin": "ArgMin",
        "slice": "Slice",
        "gather": "Gather",
        "scatter": "ScatterND",
        "cat": "Concat",
        "stack": "Concat",
        "relu": "Relu",
    }

    op_type = "Unknown"

    # Exact match first, then fallback
    if target_name in op_map:
        op_type = op_map[target_name]
    else:
        for key, val in op_map.items():
            if target_name.startswith(key + "_") or target_name.endswith("." + key):
                op_type = val
                break

    inputs = []
    for arg in node.args:
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    attributes = {}
    if hasattr(node, "kwargs"):
        for k, v in node.kwargs.items():
            if isinstance(v, (list, tuple)):
                attributes[k] = list(v)
            elif isinstance(v, (int, float, str, bool)):
                attributes[k] = v

    l_node = LogicalNode(
        id=node.name,
        op_type=op_type,
        domain="ai.onnx",
        version=1,
        attributes=attributes,
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node.name] = l_node
    node_map[node.name] = node.name


def _handle_fx_output(node, graph: LogicalGraph, node_map: dict[str, str], modules: dict = None) -> None:
    outputs = []
    args = node.args[0]
    if isinstance(args, tuple):
        outputs.extend([arg.name for arg in args if hasattr(arg, "name")])
    elif hasattr(args, "name"):
        outputs.append(args.name)
    elif isinstance(args, list):
        outputs.extend([arg.name for arg in args if hasattr(arg, "name")])
    graph.outputs = outputs


def ingest_torch_fx(fx_graph_module) -> LogicalGraph:
    """Ingest Torch FX graph module."""
    if fx_graph_module is None:
        raise ValueError("Torch FX GraphModule cannot be None")

    graph = LogicalGraph(name="torch_fx_ingested")

    if not hasattr(fx_graph_module, "graph") or not hasattr(fx_graph_module.graph, "nodes"):
        return graph

    handlers = {
        "placeholder": _handle_fx_placeholder,
        "call_function": _handle_fx_call_function,
        "call_method": _handle_fx_call_method,
        "call_module": _handle_fx_call_module,
        "get_attr": _handle_fx_get_attr,
        "output": _handle_fx_output,
    }

    modules = {}
    if hasattr(fx_graph_module, "named_modules"):
        try:
            modules = dict(fx_graph_module.named_modules())
        except Exception:
            pass

    node_map: dict[str, str] = {}
    for node in fx_graph_module.graph.nodes:
        if node.op in handlers:
            handlers[node.op](node, graph, node_map, modules)
        else:
            pass
    return graph


def _extract_jaxpr_constants(jaxpr, graph: LogicalGraph) -> None:
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


def _translate_jax_equation(eqn, graph: LogicalGraph) -> None:
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

    attributes = {}
    if hasattr(eqn, "params"):
        for k, v in eqn.params.items():
            if isinstance(v, (list, tuple)):
                attributes[k] = list(v)
            elif isinstance(v, (int, float, str, bool)):
                attributes[k] = v

    shape_metadata = ()
    if eqn.outvars and hasattr(eqn.outvars[0], "aval") and hasattr(eqn.outvars[0].aval, "shape"):
        shape_metadata = tuple(eqn.outvars[0].aval.shape)

    l_node = LogicalNode(
        id=out_id,
        op_type=op_type,
        domain="ai.onnx",
        version=1,
        attributes=attributes,
        inputs=inputs,
        shape_metadata=shape_metadata,
    )

    graph.nodes[out_id] = l_node


def ingest_jaxpr(jaxpr) -> LogicalGraph:
    """Ingest JAX jaxpr."""
    if jaxpr is None:
        raise ValueError("JAX jaxpr cannot be None")

    graph = LogicalGraph(name="jaxpr_ingested")

    _extract_jaxpr_constants(jaxpr, graph)

    if hasattr(jaxpr, "eqns"):
        for eqn in jaxpr.eqns:
            _translate_jax_equation(eqn, graph)

    return graph
