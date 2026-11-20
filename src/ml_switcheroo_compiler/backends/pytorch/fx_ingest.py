"""PyTorch FX GraphModule to LogicalGraph ingestion adapter."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.shape import OpValue, VariadicCallableProtocol


@typing.runtime_checkable
class FxNodeProtocol(typing.Protocol):
    """Protocol for Torch FX nodes exposing ``name`` and ``op``."""

    @property
    def name(self) -> str:
        """Return FX node name.

        Returns:
            str: Node identifier.
        """
        ...

    @property
    def op(self) -> str:
        """Return FX node opcode.

        Returns:
            str: FX opcode (``placeholder``, ``call_function``, ``call_method``, ``call_module``, ``get_attr``,
                ``output``).
        """
        ...


@typing.runtime_checkable
class _FxNodeArgsProtocol(typing.Protocol):
    """Protocol for FX nodes exposing positional ``args``."""

    @property
    def args(self) -> Sequence[OpValue]:
        """Return FX node positional arguments.

        Returns:
            Sequence[OpValue]: Positional arguments.
        """
        ...


@typing.runtime_checkable
class _FxNodeKwargsProtocol(typing.Protocol):
    """Protocol for FX nodes exposing keyword ``kwargs``."""

    @property
    def kwargs(self) -> Mapping[str, OpValue]:
        """Return FX node keyword arguments.

        Returns:
            Mapping[str, OpValue]: Keyword arguments.
        """
        ...


@typing.runtime_checkable
class _FxNodeTargetProtocol(typing.Protocol):
    """Protocol for FX nodes exposing ``target``."""

    @property
    def target(self) -> OpValue:
        """Return FX node target callable, method name, or module path.

        Returns:
            OpValue: Target descriptor.
        """
        ...


@typing.runtime_checkable
class _HasNameProtocol(typing.Protocol):
    """Protocol for objects exposing a ``name`` attribute."""

    @property
    def name(self) -> str:
        """Return object name.

        Returns:
            str: Object name string.
        """
        ...


@typing.runtime_checkable
class _HasDunderNameProtocol(typing.Protocol):
    """Protocol for callables exposing ``__name__``."""

    @property
    def __name__(self) -> str:
        """Return callable ``__name__``.

        Returns:
            str: Callable name string.
        """
        ...


@typing.runtime_checkable
class _FxGraphInnerProtocol(typing.Protocol):
    """Protocol for inner ``torch.fx.Graph`` exposing ``nodes``."""

    @property
    def nodes(self) -> Iterable[FxNodeProtocol]:
        """Return iterable of FX nodes.

        Returns:
            Iterable[FxNodeProtocol]: FX graph nodes.
        """
        ...


@typing.runtime_checkable
class FxGraphModuleProtocol(typing.Protocol):
    """Protocol for ``torch.fx.GraphModule`` exposing ``graph``."""

    @property
    def graph(self) -> _FxGraphInnerProtocol | OpValue:
        """Return underlying FX graph.

        Returns:
            _FxGraphInnerProtocol | OpValue: FX graph object.
        """
        ...


@typing.runtime_checkable
class _NamedModulesProtocol(typing.Protocol):
    """Protocol for ``nn.Module`` instances exposing ``named_modules()``."""

    def named_modules(self) -> Iterable[tuple[str, VariadicCallableProtocol]]:
        """Yield ``(name, module)`` pairs for registered submodules.

        Returns:
            Iterable[tuple[str, VariadicCallableProtocol]]: Named submodule pairs.
        """
        ...


def _extract_node_attributes(node: FxNodeProtocol | LogicalNode | OpValue) -> dict[str, AttributeValue]:
    """Extract serializable ``AttributeValue`` entries from an FX node's ``kwargs``.

    Args:
        node (FxNodeProtocol | LogicalNode | OpValue): Target FX node.

    Returns:
        dict[str, AttributeValue]: Extracted attribute dictionary.
    """
    attributes: dict[str, AttributeValue] = {}
    cand_node: FxNodeProtocol | LogicalNode | OpValue | _FxNodeKwargsProtocol = node
    raw_kwargs: Mapping[str, OpValue] = (
        cand_node.kwargs
        if isinstance(cand_node, _FxNodeKwargsProtocol) and isinstance(cand_node.kwargs, Mapping)
        else {}
    )
    for k, v in raw_kwargs.items():
        if isinstance(v, (list, tuple)):
            attributes[str(k)] = [x for x in v if isinstance(x, (int, float, str, bool))]
        elif isinstance(v, (int, float, str, bool)):
            attributes[str(k)] = v
    return attributes


def _handle_fx_placeholder(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Register a Torch FX placeholder node in ``node_map``.

    Args:
        node (FxNodeProtocol): Torch FX placeholder node.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.
    """
    del graph, modules
    node_map[node.name] = node.name


def _extract_fx_inputs(args: Sequence[OpValue], node_map: dict[str, str]) -> list[str]:
    """Resolve FX node positional arguments into ``LogicalNode`` input IDs.

    Args:
        args (Sequence[OpValue]): FX node positional arguments.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.

    Returns:
        list[str]: Resolved input node IDs.
    """
    inputs: list[str] = []
    for arg in args:
        cand_arg: OpValue | _HasNameProtocol = arg
        if isinstance(cand_arg, _HasNameProtocol) and cand_arg.name in node_map:
            inputs.append(node_map[cand_arg.name])
        else:
            inputs.append(str(arg))
    return inputs


def _handle_fx_call_method(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX ``call_method`` node into a ``LogicalNode``.

    Args:
        node (FxNodeProtocol): Torch FX ``call_method`` node.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.
    """
    del modules
    target_name = str(node.target) if isinstance(node, _FxNodeTargetProtocol) else ""

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
    node_args = node.args if isinstance(node, _FxNodeArgsProtocol) else ()
    inputs = _extract_fx_inputs(node_args, node_map)

    node_name = str(node.name)
    graph.nodes[node_name] = LogicalNode(
        id=node_name,
        op_type=op_type,
        attributes=_extract_node_attributes(node),
        inputs=inputs,
        shape_metadata=(),
    )
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
    """Resolve the compiler ``op_type`` for a Torch FX submodule target.

    Args:
        target_name (str): Target submodule path string.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.

    Returns:
        str: Resolved IR ``op_type`` string.
    """
    if modules is not None and target_name in modules:
        mod_type = type(modules[target_name]).__name__
        for pattern, mapped_op in _FX_MODULE_TYPE_MAP:
            if pattern in mod_type:
                return mapped_op
    return "CallModule"


def _handle_fx_call_module(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX ``call_module`` node into a ``LogicalNode``.

    Args:
        node (FxNodeProtocol): Torch FX ``call_module`` node.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.
    """
    target_name = str(node.target) if isinstance(node, _FxNodeTargetProtocol) else ""
    op_type = _resolve_fx_module_op_type(target_name, modules)
    node_args = node.args if isinstance(node, _FxNodeArgsProtocol) else ()
    inputs = _extract_fx_inputs(node_args, node_map)

    node_name = str(node.name)
    graph.nodes[node_name] = LogicalNode(
        id=node_name,
        op_type=op_type,
        attributes=_extract_node_attributes(node),
        inputs=inputs,
        shape_metadata=(),
    )
    node_map[node_name] = node_name


def _handle_fx_get_attr(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX ``get_attr`` node into a ``Constant`` ``LogicalNode``.

    Args:
        node (FxNodeProtocol): Torch FX ``get_attr`` node.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.
    """
    del modules
    target_str = str(node.target) if isinstance(node, _FxNodeTargetProtocol) else ""
    node_name = str(node.name)
    graph.nodes[node_name] = LogicalNode(
        id=node_name,
        op_type="Constant",
        attributes={"target": target_str},
        inputs=[],
        shape_metadata=(),
    )
    node_map[node_name] = node_name


def _handle_fx_call_function(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX ``call_function`` node into a ``LogicalNode``.

    Args:
        node (FxNodeProtocol): Torch FX ``call_function`` node.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.
    """
    del modules
    target_obj: OpValue = node.target if isinstance(node, _FxNodeTargetProtocol) else ""
    target_name = str(target_obj.__name__) if isinstance(target_obj, _HasDunderNameProtocol) else str(target_obj)

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

    node_args = node.args if isinstance(node, _FxNodeArgsProtocol) else ()
    inputs = _extract_fx_inputs(node_args, node_map)

    node_name = str(node.name)
    graph.nodes[node_name] = LogicalNode(
        id=node_name,
        op_type=op_type,
        attributes=_extract_node_attributes(node),
        inputs=inputs,
        shape_metadata=(),
    )
    node_map[node_name] = node_name


def _handle_fx_output(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Populate ``graph.outputs`` from a Torch FX ``output`` node.

    Args:
        node (FxNodeProtocol): Torch FX ``output`` node.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
        node_map (dict[str, str]): Mapping from FX node names to ``LogicalNode`` IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional named submodules map.
    """
    del node_map, modules
    outputs: list[str] = []
    node_args = node.args if (isinstance(node, _FxNodeArgsProtocol) and node.args) else (None,)
    args = node_args[0]
    if isinstance(args, (tuple, list)):
        for item in args:
            if isinstance(item, _HasNameProtocol):
                outputs.append(str(item.name))
    elif isinstance(args, _HasNameProtocol):
        outputs.append(str(args.name))
    graph.outputs = outputs


def ingest_torch_fx(fx_graph_module: OpValue) -> LogicalGraph:
    """Ingest a Torch FX ``GraphModule`` into a canonical ``LogicalGraph``.

    Args:
        fx_graph_module (OpValue): Torch FX ``GraphModule`` instance to convert.

    Returns:
        LogicalGraph: Converted compiler intermediate representation graph.

    Raises:
        ValueError: If ``fx_graph_module`` is ``None``.
    """
    if fx_graph_module is None:
        raise ValueError("Torch FX GraphModule cannot be None")

    graph = LogicalGraph(name="torch_fx_ingested")

    if not isinstance(fx_graph_module, FxGraphModuleProtocol) or not isinstance(
        fx_graph_module.graph, _FxGraphInnerProtocol
    ):
        return graph
    raw_inner = fx_graph_module.graph

    handlers = {
        "placeholder": _handle_fx_placeholder,
        "call_function": _handle_fx_call_function,
        "call_method": _handle_fx_call_method,
        "call_module": _handle_fx_call_module,
        "get_attr": _handle_fx_get_attr,
        "output": _handle_fx_output,
    }

    modules: dict[str, VariadicCallableProtocol] = {}
    if isinstance(fx_graph_module, _NamedModulesProtocol) and callable(fx_graph_module.named_modules):
        modules = {str(k): v for k, v in fx_graph_module.named_modules()}

    node_map: dict[str, str] = {}
    for node in raw_inner.nodes:
        if node.op in handlers:
            handlers[node.op](node, graph, node_map, modules)
    return graph


__all__ = [
    "FxGraphModuleProtocol",
    "FxNodeProtocol",
    "_FX_MODULE_TYPE_MAP",
    "_extract_fx_inputs",
    "_extract_node_attributes",
    "_handle_fx_call_function",
    "_handle_fx_call_method",
    "_handle_fx_call_module",
    "_handle_fx_get_attr",
    "_handle_fx_output",
    "_handle_fx_placeholder",
    "_resolve_fx_module_op_type",
    "ingest_torch_fx",
]
