"""Foreign module integration and external graph ingestion adapter layer."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    ShapeProtocol,
)
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar, VariadicCallableProtocol, invoke_callable
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _is_foreign_int(val: TensorOrScalar | type) -> typing.TypeGuard[int]:
    """Type guard narrowing `val` (including `int` type annotations) to `int`.

    Args:
        val (TensorOrScalar | type): Candidate dimension value.

    Returns:
        bool: True if val is an int or the `int` type.
    """
    return isinstance(val, int) or val is int


def _foreign_to_int(val: TensorOrScalar | type) -> int:
    """Coerce a dimension value to int.

    Args:
        val (TensorOrScalar | type): Raw dimension value.

    Returns:
        int: Coerced integer value.
    """
    assert _is_foreign_int(val)
    return val


def _is_foreign_shape_tuple(seq: tuple[int | str | None, ...]) -> typing.TypeGuard[tuple[int, ...]]:
    """Type guard narrowing a shape tuple to `tuple[int, ...]`.

    Args:
        seq (tuple[int | str | None, ...]): Shape tuple.

    Returns:
        bool: True if seq is a tuple.
    """
    return isinstance(seq, tuple)


def _foreign_extract_shape(seq: Sequence[int | str | None]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a concrete integer shape tuple.

    Args:
        seq (Sequence[int | str | None]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    res = tuple(seq)
    assert _is_foreign_shape_tuple(res)
    return res


@typing.runtime_checkable
class FxNodeProtocol(typing.Protocol):
    """Protocol for Torch FX Graph nodes."""

    @property
    def op(self) -> str:
        """Return FX node operation type ('placeholder', 'call_function', etc.).

        Returns:
            str: Operation kind.
        """
        ...

    @property
    def name(self) -> str:
        """Return FX node name.

        Returns:
            str: Node name.
        """
        ...


@typing.runtime_checkable
class _FxNodeKwargsProtocol(typing.Protocol):
    """Protocol for FX nodes exposing `.kwargs`."""

    @property
    def kwargs(self) -> Mapping[str, TensorOrScalar]:
        """Return FX node keyword arguments.

        Returns:
            Mapping[str, TensorOrScalar]: Keyword arguments.
        """
        ...


@typing.runtime_checkable
class _FxNodeArgsProtocol(typing.Protocol):
    """Protocol for FX nodes exposing `.args`."""

    @property
    def args(self) -> tuple[TensorOrScalar, ...]:
        """Return FX node positional arguments.

        Returns:
            tuple[TensorOrScalar, ...]: Positional arguments.
        """
        ...


@typing.runtime_checkable
class _FxNodeTargetProtocol(typing.Protocol):
    """Protocol for FX nodes exposing `.target`."""

    @property
    def target(self) -> TensorOrScalar:
        """Return FX node target callable or string.

        Returns:
            TensorOrScalar: Node target.
        """
        ...


@typing.runtime_checkable
class _FxGraphInnerProtocol(typing.Protocol):
    """Protocol for the inner `.graph` of a Torch FX GraphModule."""

    @property
    def nodes(self) -> Iterable[FxNodeProtocol]:
        """Return iterable of FX nodes.

        Returns:
            Iterable[FxNodeProtocol]: FX nodes.
        """
        ...


@typing.runtime_checkable
class FxGraphModuleProtocol(typing.Protocol):
    """Protocol for Torch FX GraphModule instances."""

    @property
    def graph(self) -> _FxGraphInnerProtocol | TensorOrScalar:
        """Return underlying FX graph.

        Returns:
            _FxGraphInnerProtocol | TensorOrScalar: FX graph.
        """
        ...


@typing.runtime_checkable
class _NamedModulesProtocol(typing.Protocol):
    """Protocol for modules exposing `named_modules()`."""

    def named_modules(self) -> Iterable[tuple[str, VariadicCallableProtocol]]:
        """Return iterable of `(name, module)` pairs.

        Returns:
            Iterable[tuple[str, VariadicCallableProtocol]]: Named submodules.
        """
        ...


@typing.runtime_checkable
class _HasNameProtocol(typing.Protocol):
    """Protocol for objects exposing a `.name` string attribute."""

    @property
    def name(self) -> str:
        """Return object name.

        Returns:
            str: Object name.
        """
        ...


@typing.runtime_checkable
class _HasDunderNameProtocol(typing.Protocol):
    """Protocol for callables exposing a `__name__` attribute."""

    @property
    def __name__(self) -> str:
        """Return callable name.

        Returns:
            str: Callable name.
        """
        ...


@typing.runtime_checkable
class _JaxAvalProtocol(typing.Protocol):
    """Protocol for JAX abstract value (`.aval`) objects."""

    @property
    def shape(self) -> Sequence[int]:
        """Return abstract value shape.

        Returns:
            Sequence[int]: Shape dimensions.
        """
        ...


@typing.runtime_checkable
class _JaxVarProtocol(typing.Protocol):
    """Protocol for JAX variable objects exposing `.aval`."""

    @property
    def aval(self) -> _JaxAvalProtocol:
        """Return abstract value.

        Returns:
            _JaxAvalProtocol: Abstract value.
        """
        ...


@typing.runtime_checkable
class JaxprEqnProtocol(typing.Protocol):
    """Protocol for JAXpr equation objects."""

    @property
    def primitive(self) -> TensorOrScalar:
        """Return JAX primitive object.

        Returns:
            TensorOrScalar: Primitive object.
        """
        ...

    @property
    def invars(self) -> Sequence[TensorOrScalar]:
        """Return input variables.

        Returns:
            Sequence[TensorOrScalar]: Input variables.
        """
        ...

    @property
    def outvars(self) -> Sequence[TensorOrScalar]:
        """Return output variables.

        Returns:
            Sequence[TensorOrScalar]: Output variables.
        """
        ...


@typing.runtime_checkable
class _JaxprEqnParamsProtocol(typing.Protocol):
    """Protocol for JAXpr equations exposing `.params`."""

    @property
    def params(self) -> Mapping[str, TensorOrScalar]:
        """Return equation parameters.

        Returns:
            Mapping[str, TensorOrScalar]: Equation parameters.
        """
        ...


@typing.runtime_checkable
class JaxprProtocol(typing.Protocol):
    """Protocol for JAXpr graph objects."""

    @property
    def eqns(self) -> Sequence[JaxprEqnProtocol]:
        """Return sequence of JAXpr equations.

        Returns:
            Sequence[JaxprEqnProtocol]: Equations.
        """
        ...


@typing.runtime_checkable
class _JaxprConstsProtocol(typing.Protocol):
    """Protocol for JAXpr objects exposing `consts`."""

    @property
    def consts(self) -> Sequence[TensorOrScalar]:
        """Return constant values.

        Returns:
            Sequence[TensorOrScalar]: Constant values.
        """
        ...


@typing.runtime_checkable
class _JaxprConstvarsProtocol(typing.Protocol):
    """Protocol for JAXpr objects exposing `constvars`."""

    @property
    def constvars(self) -> Sequence[TensorOrScalar] | None:
        """Return constant variables.

        Returns:
            Sequence[TensorOrScalar] | None: Constant variables.
        """
        ...


@typing.runtime_checkable
class KerasLayerProtocol(typing.Protocol):
    """Protocol for Keras layers exposing `output_shape`."""

    @property
    def output_shape(self) -> Sequence[int] | int:
        """Return output shape.

        Returns:
            Sequence[int] | int: Output shape.
        """
        ...


@typing.runtime_checkable
class _OutputShapesProtocol(typing.Protocol):
    """Protocol for objects exposing `output_shapes`."""

    @property
    def output_shapes(self) -> Sequence[Sequence[int]]:
        """Return output shapes list.

        Returns:
            Sequence[Sequence[int]]: Output shapes.
        """
        ...


@typing.runtime_checkable
class _SubgraphNodeMetaProtocol(typing.Protocol):
    """Protocol for subgraph nodes exposing `shape_metadata`."""

    @property
    def shape_metadata(self) -> Sequence[int | str] | None:
        """Return node shape metadata.

        Returns:
            Sequence[int | str] | None: Shape metadata.
        """
        ...


@typing.runtime_checkable
class _SubgraphOutputsProtocol(typing.Protocol):
    """Protocol for subgraph objects exposing `outputs` and `nodes`."""

    @property
    def outputs(self) -> Sequence[str]:
        """Return output node identifiers.

        Returns:
            Sequence[str]: Output IDs.
        """
        ...

    @property
    def nodes(self) -> Mapping[str, _SubgraphNodeMetaProtocol | LogicalNode]:
        """Return mapping of node IDs to nodes.

        Returns:
            Mapping[str, _SubgraphNodeMetaProtocol | LogicalNode]: Nodes mapping.
        """
        ...


@typing.runtime_checkable
class _SubgraphHolderProtocol(typing.Protocol):
    """Protocol for objects exposing a `subgraph` attribute."""

    @property
    def subgraph(self) -> _SubgraphOutputsProtocol:
        """Return attached LogicalGraph subgraph.

        Returns:
            _SubgraphOutputsProtocol: Attached subgraph.
        """
        ...


@typing.runtime_checkable
class _GenericAliasArgsProtocol(typing.Protocol):
    """Protocol for typing annotations exposing `__args__`."""

    @property
    def __args__(self) -> tuple[int | type | str, ...]:
        """Return generic type arguments.

        Returns:
            tuple[int | type | str, ...]: Type arguments.
        """
        ...


def _infer_shape_from_subgraph(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Extract output shape metadata from an attached subgraph's output nodes.

    Args:
        first (TensorOrScalar): Target object potentially containing a subgraph attribute.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if found, else None.
    """
    raw_first: TensorOrScalar = first
    if isinstance(raw_first, _SubgraphHolderProtocol):
        sub = raw_first.subgraph
        out_nodes = [sub.nodes.get(out) for out in sub.outputs if out in sub.nodes]
        first_node = out_nodes[0] if out_nodes else None
        if first_node is not None and isinstance(first_node, _SubgraphNodeMetaProtocol) and first_node.shape_metadata:
            return _foreign_extract_shape(first_node.shape_metadata)
    return None


def _infer_shape_from_attributes(first: TensorOrScalar) -> tuple[int, ...] | None:
    """Extract output shape from common tensor or module shape attributes.

    Args:
        first (TensorOrScalar): Target object or module to inspect.

    Returns:
        tuple[int, ...] | None: Inferred shape tuple if present, else None.
    """
    raw_first: TensorOrScalar | KerasLayerProtocol | _OutputShapesProtocol = first
    if isinstance(raw_first, ShapeProtocol):
        return _foreign_extract_shape(raw_first.shape)
    if isinstance(raw_first, KerasLayerProtocol):
        s = raw_first.output_shape
        return tuple(int(d) for d in s) if isinstance(s, (list, tuple)) else (_foreign_to_int(s),)
    if isinstance(raw_first, _OutputShapesProtocol):
        s_list = raw_first.output_shapes
        return tuple(int(d) for d in s_list[0]) if isinstance(s_list, (list, tuple)) and len(s_list) > 0 else ()
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
        hints = invoke_callable(typing.get_type_hints, first)
        if isinstance(hints, dict) and "return" in hints:
            args_tuple = typing.get_args(hints["return"])
            if args_tuple and _is_int_type_sequence(args_tuple):
                return tuple(_foreign_to_int(x) for x in args_tuple if isinstance(x, int) or x is int)
    except (TypeError, NameError) as e:
        import logging

        logging.getLogger(__name__).warning("Failed to infer return type hints for %s: %s", first, e)

    try:
        sig = invoke_callable(inspect.signature, first)
        raw_sig: TensorOrScalar = sig
        ret_anno: TensorOrScalar = raw_sig.return_annotation if isinstance(raw_sig, _SignatureReturnAnnotationProtocol) else None
        if isinstance(ret_anno, _GenericAliasArgsProtocol) and _is_int_type_sequence(tuple(ret_anno.__args__)):
            return tuple(_foreign_to_int(x) for x in tuple(ret_anno.__args__) if isinstance(x, int) or x is int)
    except (ValueError, TypeError) as e:
        from ml_switcheroo_compiler.core.errors import ShapeInferenceError

        raise ShapeInferenceError(f"Failed to infer shape from signature for {first}: {e}") from e

    return None


@typing.runtime_checkable
class _SignatureReturnAnnotationProtocol(typing.Protocol):
    """Protocol for `inspect.Signature` exposing `return_annotation`."""

    @property
    def return_annotation(self) -> TensorOrScalar:
        """Return signature return annotation.

        Returns:
            TensorOrScalar: Return annotation.
        """
        ...


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
        raw_src = invoke_callable(inspect.getsource, first)
        src_str = raw_src if isinstance(raw_src, str) else str(raw_src)
        lines = src_str.splitlines()
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
                return tuple(_foreign_to_int(d) for d in shape_val)
            return (_foreign_to_int(shape_val),)

        if args:
            first = args[0]
            attr_shape = _infer_shape_from_attributes(first)
            if attr_shape is not None:
                return attr_shape

            if callable(first):
                call_shape = _infer_shape_from_callable(first)
                if call_shape is not None:
                    return call_shape

            if len(args) > 1:
                second: TensorOrScalar = args[1]
                if isinstance(second, ShapeProtocol):
                    return _foreign_extract_shape(second.shape)

        return ()


def _extract_node_attributes(node: FxNodeProtocol | LogicalNode) -> dict[str, AttributeValue]:
    """Extract clean AttributeValue dictionary from an FX node's kwargs.

    Args:
        node (FxNodeProtocol | LogicalNode): Target FX node.

    Returns:
        dict[str, AttributeValue]: Cleaned attribute mapping.
    """
    attributes: dict[str, AttributeValue] = {}
    raw_node: FxNodeProtocol | LogicalNode | TensorOrScalar = node
    raw_kwargs: Mapping[str, TensorOrScalar] = raw_node.kwargs if isinstance(raw_node, _FxNodeKwargsProtocol) else {}
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
    """Register a Torch FX placeholder node in the node mapping.

    Args:
        node (FxNodeProtocol): The Torch FX node representing an input placeholder.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del graph, modules
    node_map[node.name] = node.name


def _extract_fx_inputs(args: Sequence[TensorOrScalar], node_map: dict[str, str]) -> list[str]:
    """Resolve FX node positional arguments into a list of input node IDs.

    Args:
        args (Sequence[TensorOrScalar]): FX node arguments.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.

    Returns:
        list[str]: Resolved input node IDs.
    """
    inputs: list[str] = []
    for arg in args:
        raw_arg: TensorOrScalar = arg
        if isinstance(raw_arg, _HasNameProtocol) and raw_arg.name in node_map:
            inputs.append(node_map[raw_arg.name])
        else:
            inputs.append(str(arg))
    return inputs


def _handle_fx_call_method(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX call_method node into a LogicalNode.

    Args:
        node (FxNodeProtocol): The Torch FX call_method node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del modules
    raw_node: FxNodeProtocol | TensorOrScalar = node
    target_name = str(raw_node.target) if isinstance(raw_node, _FxNodeTargetProtocol) else ""

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
    node_args = raw_node.args if isinstance(raw_node, _FxNodeArgsProtocol) else ()
    inputs = _extract_fx_inputs(node_args, node_map)

    node_name = str(node.name)
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
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX call_module node into a LogicalNode.

    Args:
        node (FxNodeProtocol): The Torch FX call_module node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    raw_node: FxNodeProtocol | TensorOrScalar = node
    target_name = str(raw_node.target) if isinstance(raw_node, _FxNodeTargetProtocol) else ""
    op_type = _resolve_fx_module_op_type(target_name, modules)
    node_args = raw_node.args if isinstance(raw_node, _FxNodeArgsProtocol) else ()
    inputs = _extract_fx_inputs(node_args, node_map)

    node_name = str(node.name)
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
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX get_attr node into a Constant LogicalNode.

    Args:
        node (FxNodeProtocol): The Torch FX get_attr node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del modules
    raw_node: FxNodeProtocol | TensorOrScalar = node
    target_str = str(raw_node.target) if isinstance(raw_node, _FxNodeTargetProtocol) else ""
    node_name = str(node.name)
    l_node = LogicalNode(
        id=node_name,
        op_type="Constant",
        attributes={"target": target_str},
        inputs=[],
        shape_metadata=(),
    )
    graph.nodes[node_name] = l_node
    node_map[node_name] = node_name


def _handle_fx_call_function(
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Translate a Torch FX call_function node into a LogicalNode.

    Args:
        node (FxNodeProtocol): The Torch FX call_function node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del modules
    raw_node: FxNodeProtocol | TensorOrScalar = node
    target_obj: TensorOrScalar = raw_node.target if isinstance(raw_node, _FxNodeTargetProtocol) else ""
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

    node_args = raw_node.args if isinstance(raw_node, _FxNodeArgsProtocol) else ()
    inputs = _extract_fx_inputs(node_args, node_map)

    node_name = str(node.name)
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
    node: FxNodeProtocol,
    graph: LogicalGraph,
    node_map: dict[str, str],
    modules: dict[str, VariadicCallableProtocol] | None = None,
) -> None:
    """Populate graph outputs from a Torch FX output node.

    Args:
        node (FxNodeProtocol): The Torch FX output node.
        graph (LogicalGraph): Target LogicalGraph being populated.
        node_map (dict[str, str]): Mapping from FX node names to LogicalNode IDs.
        modules (dict[str, VariadicCallableProtocol] | None): Optional mapping of named submodules.
    """
    del node_map, modules
    outputs: list[str] = []
    raw_node: FxNodeProtocol | TensorOrScalar = node
    node_args = raw_node.args if (isinstance(raw_node, _FxNodeArgsProtocol) and raw_node.args) else (None,)
    args = node_args[0]
    raw_args: TensorOrScalar = args
    if isinstance(args, (tuple, list)):
        for arg in args:
            raw_item: TensorOrScalar = arg
            if isinstance(raw_item, _HasNameProtocol):
                outputs.append(str(raw_item.name))
    elif isinstance(raw_args, _HasNameProtocol):
        outputs.append(str(raw_args.name))
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

    raw_gm: TensorOrScalar = fx_graph_module
    if not isinstance(raw_gm, FxGraphModuleProtocol) or not isinstance(raw_gm.graph, _FxGraphInnerProtocol):
        return graph
    raw_inner = raw_gm.graph

    handlers = {
        "placeholder": _handle_fx_placeholder,
        "call_function": _handle_fx_call_function,
        "call_method": _handle_fx_call_method,
        "call_module": _handle_fx_call_module,
        "get_attr": _handle_fx_get_attr,
        "output": _handle_fx_output,
    }

    modules: dict[str, VariadicCallableProtocol] = {}
    if isinstance(raw_gm, _NamedModulesProtocol):
        modules = {str(k): v for k, v in raw_gm.named_modules()}

    node_map: dict[str, str] = {}
    for node in raw_inner.nodes:
        if node.op in handlers:
            handlers[node.op](node, graph, node_map, modules)
    return graph


def _extract_jaxpr_constants(jaxpr: TensorOrScalar, graph: LogicalGraph) -> None:
    """Extract constant values from a JAX jaxpr into Constant LogicalNodes.

    Args:
        jaxpr (TensorOrScalar): The JAX jaxpr object containing constants.
        graph (LogicalGraph): Target LogicalGraph being populated.
    """
    raw_jaxpr: TensorOrScalar = jaxpr
    constvars = jaxpr.constvars if isinstance(jaxpr, _JaxprConstvarsProtocol) else None
    if isinstance(raw_jaxpr, _JaxprConstsProtocol):
        for i, const_val in enumerate(raw_jaxpr.consts):
            const_id = str(id(constvars[i])) if constvars else f"const_{i}"
            c_shape = _foreign_extract_shape(const_val.shape) if isinstance(const_val, ShapeProtocol) else ()
            c_attr: AttributeValue = const_val if isinstance(const_val, (int, float, str, bool, bytes)) or const_val is None else str(const_val)
            l_node = LogicalNode(
                id=const_id,
                op_type="Constant",
                domain="ai.onnx",
                version=1,
                attributes={"value": c_attr},
                inputs=[],
                shape_metadata=c_shape,
            )
            graph.nodes[const_id] = l_node


def _translate_jax_equation(eqn: JaxprEqnProtocol | TensorOrScalar, graph: LogicalGraph) -> None:
    """Translate a single JAXpr equation into a LogicalNode.

    Args:
        eqn (JaxprEqnProtocol | TensorOrScalar): The JAXpr equation object.
        graph (LogicalGraph): Target LogicalGraph being populated.
    """
    raw_eqn: JaxprEqnProtocol | TensorOrScalar = eqn
    op_type = "Unknown"
    has_eqn_attrs = isinstance(raw_eqn, JaxprEqnProtocol)
    primitive_obj: TensorOrScalar = raw_eqn.primitive if isinstance(raw_eqn, JaxprEqnProtocol) else ""
    primitive_name = str(primitive_obj.name) if isinstance(primitive_obj, _HasNameProtocol) else str(primitive_obj)
    if primitive_name == "add":
        op_type = "Add"
    elif primitive_name == "mul":
        op_type = "Mul"

    invars = raw_eqn.invars if isinstance(raw_eqn, JaxprEqnProtocol) else ()
    inputs = [str(id(invar)) for invar in invars]

    outvars = list(raw_eqn.outvars) if has_eqn_attrs and isinstance(raw_eqn, JaxprEqnProtocol) else []
    out_id = str(id(outvars[0])) if outvars else "out"

    attributes: dict[str, AttributeValue] = {}
    if isinstance(raw_eqn, _JaxprEqnParamsProtocol) and isinstance(raw_eqn.params, Mapping):
        for k, v in raw_eqn.params.items():
            if isinstance(v, (list, tuple)):
                attributes[str(k)] = [x for x in v if isinstance(x, (int, float, str, bool))]
            elif isinstance(v, (int, float, str, bool)):
                attributes[str(k)] = v

    shape_metadata: tuple[int, ...] = ()
    if outvars:
        first_out: TensorOrScalar | _JaxVarProtocol = outvars[0]
        if isinstance(first_out, _JaxVarProtocol):
            shape_metadata = tuple(int(d) for d in first_out.aval.shape)

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

    raw_jaxpr: TensorOrScalar = jaxpr
    if isinstance(raw_jaxpr, JaxprProtocol):
        for eqn in raw_jaxpr.eqns:
            _translate_jax_equation(eqn, graph)

    return graph
