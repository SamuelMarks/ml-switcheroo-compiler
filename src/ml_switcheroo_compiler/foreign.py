"""Foreign module integration and external graph ingestion adapter layer."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import as_callable, as_int, as_str
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import VariadicCallableProtocol


def _infer_shape_from_subgraph(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Extract output shape metadata from an attached subgraph's output nodes.

    Args:
        first (TensorOrScalar): Target object potentially containing a subgraph attribute.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if found, else None.
    """
    if hasattr(first, "subgraph") and hasattr(getattr(first, "subgraph"), "outputs"):
        out_nodes = [getattr(first, "subgraph").nodes.get(out) for out in getattr(first, "subgraph").outputs if out in getattr(first, "subgraph").nodes]
        if out_nodes and hasattr(out_nodes[0], "shape_metadata") and out_nodes[0].shape_metadata:
            return tuple(out_nodes[0].shape_metadata)
    return None


def _infer_shape_from_attributes(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Extract output shape from common tensor or module shape attributes.

    Args:
        first (TensorOrScalar): Target object or module to inspect.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if present, else None.
    """
    if hasattr(first, "shape"):
        return tuple(getattr(first, "shape"))
    if hasattr(first, "output_shape"):
        s = getattr(first, "output_shape")
        return tuple(s) if isinstance(s, (list, tuple)) else (s,)
    if hasattr(first, "output_shapes"):
        s = getattr(first, "output_shapes")
        return tuple(s[0]) if isinstance(s, (list, tuple)) and len(s) > 0 else ()
    return _infer_shape_from_subgraph(first)


def _is_int_type_sequence(seq: tuple[int | type | str, ...]) -> bool:
    """Check whether all elements in a type argument tuple are integers or the int type.

    Args:
        seq (tuple[int | type | str, ...]): Sequence of type arguments.

    Returns:
        bool: True if all elements are int instances or the int type.
    """
    return all(isinstance(x, int) or x is int for x in seq)


def _infer_shape_from_callable_annotations(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Infer output shape from callable return type hints or signature annotations.

    Args:
        first (TensorOrScalar): Target callable to inspect.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if found, else None.

    Raises:
        ShapeInferenceError: If inspecting the callable signature fails.
    """
    import inspect

    try:
        hints = as_callable(typing.get_type_hints)(first)
        if "return" in hints:
            args_tuple = typing.get_args(hints["return"])
            if args_tuple and _is_int_type_sequence(args_tuple):
                return tuple(as_int(x) for x in args_tuple if isinstance(x, int) or x is int)
    except (TypeError, NameError) as e:
        import logging

        logging.getLogger(__name__).warning("Failed to infer return type hints for %s: %s", first, e)

    try:
        sig = as_callable(inspect.signature)(first)
        ret_anno = sig.return_annotation
        if hasattr(ret_anno, "__args__") and _is_int_type_sequence(tuple(getattr(ret_anno, "__args__"))):
            return tuple(as_int(x) for x in tuple(getattr(ret_anno, "__args__")) if isinstance(x, int) or x is int)
    except (ValueError, TypeError) as e:
        from ml_switcheroo_compiler.core.errors import ShapeInferenceError

        raise ShapeInferenceError(f"Failed to infer shape from signature for {first}: {e}") from e

    return None


def _extract_return_tuple_dims(src_code: str) -> tuple[int, ...] | None:
    """Parse source code AST and extract constant integer tuple dimensions from Return nodes.

    Args:
        src_code (str): Dedented Python source code string.

    Returns:
        tuple[int, ...] | None: Tuple of integer dimensions if found, else None.
    """
    import ast

    tree = ast.parse(src_code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple):
            dims = [elt.value for elt in n.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, int)]
            if dims and len(dims) == len(n.value.elts):
                return tuple(dims)
    return None


def _infer_shape_from_callable_ast(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Infer output shape by parsing the callable's source AST for constant tuple returns.

    Args:
        first (TensorOrScalar): Target callable to inspect.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if found, else None.
    """
    import inspect

    try:
        lines = as_str(as_callable(inspect.getsource)(first)).splitlines()
        spaces = (len(lines[0]) - len(lines[0].lstrip())) if lines else 0
        src_code = chr(10).join(line[spaces:] if len(line) >= spaces else line for line in lines)
        return _extract_return_tuple_dims(src_code)
    except (OSError, TypeError, SyntaxError) as e:
        import logging

        logging.getLogger(__name__).debug("AST introspection failed for %s: %s", first, e)
        return None


def _infer_shape_from_callable(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Infer output shape from a callable via annotations or AST introspection.

    Args:
        first (TensorOrScalar): Target callable to inspect.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if found, else None.
    """
    anno_shape = _infer_shape_from_callable_annotations(first)
    if anno_shape is not None:
        return anno_shape
    return _infer_shape_from_callable_ast(first)


@register_op("ForeignCall")
class ForeignCall(OpDef):
    """Universal ForeignCall op for external graphs/modules."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: AttrValue) -> tuple[int, ...]:
        """Infer the shape of the output for ForeignCall via introspection.

        Args:
            *args (TensorOrScalar): Positional arguments (callable, inputs, ...).
            **kwargs (AttrValue): Keyword arguments including optional 'output_shape'.

        Returns:
            tuple[int, ...]: The inferred shape.
        """
        if "output_shape" in kwargs and kwargs["output_shape"] is not None:
            shape_val = kwargs["output_shape"]
            if isinstance(shape_val, (list, tuple)):
                return tuple(int(as_int(d)) for d in shape_val)
            return (int(as_int(shape_val)),)

        if args:
            first = args[0]
            attr_shape = _infer_shape_from_attributes(first)
            if attr_shape is not None:
                return attr_shape

            if callable(first):
                call_shape = _infer_shape_from_callable(first)
                if call_shape is not None:
                    return call_shape

            if len(args) > 1 and hasattr(args[1], "shape"):
                return tuple(getattr(args[1], "shape"))

        return ()


def _extract_node_attributes(node: LogicalNode) -> dict[str, AttributeValue]:
    """Extract clean AttributeValue dictionary from an FX node's kwargs.

    Args:
        node (LogicalNode): Target FX node.

    Returns:
        dict[str, AttributeValue]: Cleaned attribute mapping.
    """
    attributes: dict[str, AttributeValue] = {}
    for k, v in getattr(node, "kwargs", {}).items():
        if isinstance(v, (list, tuple)):
            attributes[str(k)] = [x for x in v if isinstance(x, (int, float, str, bool))]
        elif isinstance(v, (int, float, str, bool)):
            attributes[str(k)] = v
    return attributes


def _handle_fx_placeholder(
    node: LogicalNode,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Register a Torch FX placeholder node in the node mapping.

    Args:
        node (LogicalNode): The Torch FX node representing an input placeholder.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del graph, modules
    node_name = getattr(node, "name", "")
    node_map[node_name] = node_name


def _handle_fx_call_method(
    node: LogicalNode,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX call_method node into a LogicalNode.

    Args:
        node (LogicalNode): The Torch FX call_method node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del modules
    target_name = str(getattr(node, "target", ""))

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
    for arg in getattr(node, "args", ()):
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    node_name = str(getattr(node, "name", ""))
    l_node = LogicalNode(
        id=node_name,
        op_type=op_type,
        attributes=_extract_node_attributes(node),
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node_name] = l_node
    node_map[node_name] = node_name


_FX_MODULE_TYPE_MAP: tuple[tuple[str, str], ...] = (
    ("Conv", "Conv"),
    ("BatchNorm", "BatchNormalization"),
    ("LayerNorm", "LayerNormalization"),
    ("GroupNorm", "GroupNormalization"),
    ("Linear", "Gemm"),
    ("Dropout", "Dropout"),
    ("ReLU", "Relu"),
    ("MaxPool", "MaxPool"),
    ("AvgPool", "AveragePool"),
)


def _resolve_fx_module_op_type(target_name: str, modules: dict[str, VariadicCallableProtocol] | None) -> str:
    """Resolve the compiler op_type for a Torch FX submodule target.

    Args:
        target_name (str): The target submodule path string.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.

    Returns:
        str: Resolved IR op_type string.
    """
    if modules is not None and target_name in modules:
        mod_type = type(modules[target_name]).__name__
        for pattern, mapped_op in _FX_MODULE_TYPE_MAP:
            if pattern in mod_type:
                return mapped_op
    return "CallModule"


def _handle_fx_call_module(
    node: LogicalNode,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX call_module node into a LogicalNode.

    Args:
        node (LogicalNode): The Torch FX call_module node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    target_name = str(getattr(node, "target", ""))
    op_type = _resolve_fx_module_op_type(target_name, modules)

    inputs = []
    for arg in getattr(node, "args", ()):
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    node_name = getattr(node, "name", "")
    l_node = LogicalNode(
        id=node_name,
        op_type=op_type,
        attributes=_extract_node_attributes(node),
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node_name] = l_node
    node_map[node_name] = node_name


def _handle_fx_get_attr(
    node: LogicalNode,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX get_attr node into a Constant LogicalNode.

    Args:
        node (LogicalNode): The Torch FX get_attr node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del modules
    node_name = getattr(node, "name", "")
    l_node = LogicalNode(
        id=node_name,
        op_type="Constant",
        attributes={"target": str(getattr(node, "target", ""))},
        inputs=[],
        shape_metadata=(),
    )
    graph.nodes[node_name] = l_node
    node_map[node_name] = node_name


def _handle_fx_call_function(
    node: LogicalNode,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX call_function node into a LogicalNode.

    Args:
        node (LogicalNode): The Torch FX call_function node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del modules
    target_obj = getattr(node, "target", "")
    target_name = getattr(target_obj, "__name__", str(target_obj))

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

    if target_name in op_map:
        op_type = op_map[target_name]
    else:
        for key, val in op_map.items():
            if target_name.startswith(key + "_") or target_name.endswith("." + key):
                op_type = val
                break

    inputs = []
    for arg in getattr(node, "args", ()):
        if hasattr(arg, "name") and arg.name in node_map:
            inputs.append(node_map[arg.name])
        else:
            inputs.append(str(arg))

    node_name = str(getattr(node, "name", ""))
    l_node = LogicalNode(
        id=node_name,
        op_type=op_type,
        attributes=_extract_node_attributes(node),
        inputs=inputs,
        shape_metadata=(),
    )
    graph.nodes[node_name] = l_node
    node_map[node_name] = node_name


def _handle_fx_output(
    node: LogicalNode,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Populate graph outputs from a Torch FX output node.

    Args:
        node (LogicalNode): The Torch FX output node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del node_map, modules
    outputs: list[str] = []
    node_args = getattr(node, "args", (None,))
    args = node_args[0]
    if isinstance(args, tuple):
        outputs.extend([str(arg.name) for arg in args if hasattr(arg, "name")])
    elif hasattr(args, "name"):
        outputs.append(str(getattr(args, "name")))
    elif isinstance(args, list):
        outputs.extend([str(arg.name) for arg in args if hasattr(arg, "name")])
    graph.outputs = outputs


def ingest_torch_fx(fx_graph_module: TensorOrScalar) -> LogicalGraph:
    """Ingest a Torch FX GraphModule into a LogicalGraph.

    Args:
        fx_graph_module (TensorOrScalar): The Torch FX GraphModule instance to convert.

    Returns:
        LogicalGraph: Converted compiler intermediate representation graph.

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
        "call_method": _handle_fx_call_method,
        "call_module": _handle_fx_call_module,
        "get_attr": _handle_fx_get_attr,
        "output": _handle_fx_output,
    }

    modules: dict[str, VariadicCallableProtocol] = {}
    if hasattr(fx_graph_module, "named_modules"):
        modules = {str(k): typing.cast("VariadicCallableProtocol", v) for k, v in getattr(fx_graph_module, "named_modules")()}

    node_map: dict[str, str] = {}
    for node in getattr(fx_graph_module, "graph").nodes:
        if node.op in handlers:
            handlers[node.op](node, graph, node_map, modules)
    return graph


def _extract_jaxpr_constants(jaxpr: TensorOrScalar, graph: LogicalGraph) -> None:
    """Extract constant values from a JAX jaxpr into Constant LogicalNodes.

    Args:
        jaxpr (TensorOrScalar): The JAX jaxpr object containing constants.
        graph (LogicalGraph): Target LogicalGraph being populated.
    """
    if hasattr(jaxpr, "consts"):
        constvars = getattr(jaxpr, "constvars", None)
        for i, const_val in enumerate(getattr(jaxpr, "consts")):
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


def _translate_jax_equation(eqn: TensorOrScalar, graph: LogicalGraph) -> None:
    """Translate a single JAXpr equation into a LogicalNode.

    Args:
        eqn (TensorOrScalar): The JAXpr equation object.
        graph (LogicalGraph): Target LogicalGraph being populated.
    """
    op_type = "Unknown"
    primitive_obj = getattr(eqn, "primitive", "")
    primitive_name = getattr(primitive_obj, "name", str(primitive_obj))
    if primitive_name == "add":
        op_type = "Add"
    elif primitive_name == "mul":
        op_type = "Mul"

    inputs = []
    for invar in getattr(eqn, "invars", []):
        inputs.append(str(id(invar)))

    outvars = list(getattr(eqn, "outvars", []))
    out_id = str(id(outvars[0])) if outvars else "out"

    attributes: dict[str, AttributeValue] = {}
    if hasattr(eqn, "params"):
        for k, v in getattr(eqn, "params").items():
            if isinstance(v, (list, tuple)):
                attributes[k] = list(v)
            elif isinstance(v, (int, float, str, bool)):
                attributes[k] = v

    shape_metadata: tuple[int, ...] = ()
    if outvars and hasattr(outvars[0], "aval") and hasattr(outvars[0].aval, "shape"):
        shape_metadata = tuple(outvars[0].aval.shape)

    l_node = LogicalNode(
        id=out_id,
        op_type=op_type,
        attributes=attributes,
        inputs=inputs,
        shape_metadata=shape_metadata,
    )

    graph.nodes[out_id] = l_node


def ingest_jaxpr(jaxpr: TensorOrScalar) -> LogicalGraph:
    """Ingest a JAX jaxpr into a LogicalGraph.

    Args:
        jaxpr (TensorOrScalar): The JAX jaxpr object to convert.

    Returns:
        LogicalGraph: Converted compiler intermediate representation graph.

    Raises:
        ValueError: If jaxpr is None.
    """
    if jaxpr is None:
        raise ValueError("JAX jaxpr cannot be None")

    graph = LogicalGraph(name="jaxpr_ingested")

    _extract_jaxpr_constants(jaxpr, graph)

    if hasattr(jaxpr, "eqns"):
        for eqn in jaxpr.eqns:
            _translate_jaxpr_eqn = _translate_jax_equation
            _translate_jaxpr_eqn(eqn, graph)

    return graph
