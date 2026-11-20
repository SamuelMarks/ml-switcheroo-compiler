"""Universal ForeignCall operation definition and dynamic shape introspection."""

from __future__ import annotations

import ast
import inspect
import logging
import types
import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.errors import ShapeInferenceError
from ml_switcheroo_compiler.core.shape import IRAttrValue, OpValue, ShapeProtocol, extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@typing.runtime_checkable
class OutputShapeHolderProtocol(typing.Protocol):
    """Protocol for external callable or layer objects exposing ``output_shape``."""

    @property
    def output_shape(self) -> Sequence[int] | int:
        """Return output shape tuple or scalar dimension.

        Returns:
            Sequence[int] | int: Output shape metadata.
        """
        ...


@typing.runtime_checkable
class _OutputShapesHolderProtocol(typing.Protocol):
    """Protocol for external objects exposing ``output_shapes``."""

    @property
    def output_shapes(self) -> Sequence[Sequence[int]] | int:
        """Return sequence of output shape tuples.

        Returns:
            Sequence[Sequence[int]] | int: Output shapes metadata.
        """
        ...


@typing.runtime_checkable
class _SubgraphNodeMetaProtocol(typing.Protocol):
    """Protocol for subgraph nodes exposing ``shape_metadata``."""

    @property
    def shape_metadata(self) -> Sequence[int | str] | None:
        """Return node shape metadata.

        Returns:
            Sequence[int | str] | None: Shape metadata tuple.
        """
        ...


@typing.runtime_checkable
class _SubgraphOutputsProtocol(typing.Protocol):
    """Protocol for subgraph objects exposing ``outputs`` and ``nodes``."""

    @property
    def outputs(self) -> Sequence[str]:
        """Return output node identifiers.

        Returns:
            Sequence[str]: Output node IDs.
        """
        ...

    @property
    def nodes(self) -> Mapping[str, _SubgraphNodeMetaProtocol | LogicalNode | OpValue]:
        """Return mapping of node IDs to nodes.

        Returns:
            Mapping[str, _SubgraphNodeMetaProtocol | LogicalNode | OpValue]: Nodes mapping.
        """
        ...


@typing.runtime_checkable
class _SubgraphHolderProtocol(typing.Protocol):
    """Protocol for objects exposing an attached ``subgraph``."""

    @property
    def subgraph(self) -> _SubgraphOutputsProtocol:
        """Return attached subgraph instance.

        Returns:
            _SubgraphOutputsProtocol: Attached subgraph.
        """
        ...


@typing.runtime_checkable
class _GenericAliasArgsProtocol(typing.Protocol):
    """Protocol for generic return annotations exposing ``__args__``."""

    @property
    def __args__(self) -> tuple[OpValue, ...]:
        """Return generic type arguments.

        Returns:
            tuple[OpValue, ...]: Generic type argument tuple.
        """
        ...


def _coerce_foreign_dim(val: OpValue) -> int | type[int]:
    """Coerce a dimension descriptor into an integer or ``int`` type marker.

    Args:
        val (OpValue): Candidate dimension descriptor.

    Returns:
        int | type[int]: Concrete integer dimension or ``int`` class marker.
    """
    if val is int:
        return int
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        return int(val) if val.lstrip("-").isdigit() else 0
    return 0


def _extract_foreign_shape(seq: Sequence[OpValue]) -> tuple[int | type[int], ...]:
    """Extract a normalized dimension tuple from a raw shape sequence.

    Args:
        seq (Sequence[OpValue]): Raw shape sequence.

    Returns:
        tuple[int | type[int], ...]: Normalized tuple of dimension values.
    """
    return tuple(_coerce_foreign_dim(d) for d in seq)


@typing.runtime_checkable
class _ForeignCallableProtocol(typing.Protocol):
    """Runtime-checkable protocol for callable objects."""

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Invoke foreign callable.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            OpValue: Output value.
        """
        ...


ForeignTarget = typing.Union[
    OpValue,
    OutputShapeHolderProtocol,
    _OutputShapesHolderProtocol,
    "_SubgraphHolderProtocol",
    _ForeignCallableProtocol,
]
ForeignCallable = typing.Union[
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
    "_ForeignCallableProtocol",
]


def _infer_shape_from_subgraph(first: ForeignTarget) -> tuple[int | type[int], ...] | None:
    """Extract output shape metadata from an attached subgraph's output nodes.

    Args:
        first (ForeignTarget): Target object potentially containing a ``subgraph`` attribute.

    Returns:
        tuple[int | type[int], ...] | None: Inferred shape tuple if present, else ``None``.
    """
    if isinstance(first, _SubgraphHolderProtocol):
        sub = first.subgraph
        out_nodes = [sub.nodes.get(out) for out in sub.outputs if out in sub.nodes]
        first_node = out_nodes[0] if out_nodes else None
        if isinstance(first_node, _SubgraphNodeMetaProtocol) and first_node.shape_metadata:
            return _extract_foreign_shape(first_node.shape_metadata)
    return None


def _infer_shape_from_attributes(first: ForeignTarget) -> tuple[int | type[int], ...] | None:
    """Extract output shape from common tensor, module, or subgraph attributes.

    Args:
        first (ForeignTarget): Target object or module to inspect.

    Returns:
        tuple[int | type[int], ...] | None: Inferred shape tuple if present, else ``None``.
    """
    if isinstance(first, ShapeProtocol):
        return _extract_foreign_shape(first.shape)
    if isinstance(first, OutputShapeHolderProtocol):
        s = first.output_shape
        return tuple(int(d) for d in s) if isinstance(s, (list, tuple)) else (_coerce_foreign_dim(s),)
    if isinstance(first, _OutputShapesHolderProtocol):
        s_list = first.output_shapes
        if isinstance(s_list, (list, tuple)) and len(s_list) > 0 and isinstance(s_list[0], (list, tuple)):
            return tuple(int(d) for d in s_list[0])
        return ()
    return _infer_shape_from_subgraph(first)


def _extract_int_type_dims(seq: Sequence[OpValue]) -> tuple[int | type[int], ...] | None:
    """Extract dimension tuple when every element is an ``int`` literal or the ``int`` type.

    Args:
        seq (Sequence[OpValue]): Generic type argument sequence.

    Returns:
        tuple[int | type[int], ...] | None: Extracted tuple if non-empty and all elements are int/type[int], else
            ``None``.
    """
    if not seq:
        return None
    dims: list[int | type[int]] = []
    for item in seq:
        if item is int:
            dims.append(int)
        elif isinstance(item, int) and not isinstance(item, bool):
            dims.append(item)
        else:
            return None
    return tuple(dims)


def _infer_shape_from_callable_annotations(
    first: ForeignCallable,
) -> tuple[int | type[int], ...] | None:
    """Infer output shape from callable return type hints or signature annotations.

    Args:
        first (ForeignCallable): Target callable to inspect.

    Returns:
        tuple[int | type[int], ...] | None: Inferred shape tuple if found, else ``None``.

    Raises:
        ShapeInferenceError: If inspecting the callable signature raises ``ValueError`` or ``TypeError``.
    """
    try:
        hints = typing.get_type_hints(first)
        if "return" in hints:
            hint_dims = _extract_int_type_dims(typing.get_args(hints["return"]))
            if hint_dims is not None:
                return hint_dims
    except (TypeError, NameError) as exc:
        logging.getLogger(__name__).warning("Failed to infer return type hints for %s: %s", first, exc)

    try:
        sig = inspect.signature(first)
        ret_anno: OpValue | _GenericAliasArgsProtocol = sig.return_annotation
        if isinstance(ret_anno, _GenericAliasArgsProtocol):
            sig_dims = _extract_int_type_dims(ret_anno.__args__)
            if sig_dims is not None:
                return sig_dims
    except (ValueError, TypeError) as exc:
        raise ShapeInferenceError(f"Failed to infer shape from signature for {first}: {exc}") from exc

    return None


def _extract_return_tuple_dims(src_code: str) -> tuple[int | type[int], ...] | None:
    """Parse source code AST and extract constant integer tuple dimensions from Return nodes.

    Args:
        src_code (str): Dedented Python source code string.

    Returns:
        tuple[int | type[int], ...] | None: Tuple of integer dimensions if found, else ``None``.
    """
    tree = ast.parse(src_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            dims = [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int) and not isinstance(elt.value, bool)
            ]
            if dims and len(dims) == len(node.value.elts):
                return tuple(dims)
    return None


def _infer_shape_from_callable_ast(
    first: ForeignCallable,
) -> tuple[int | type[int], ...] | None:
    """Infer output shape by parsing the callable's source AST for constant tuple returns.

    Args:
        first (ForeignCallable): Target callable to inspect.

    Returns:
        tuple[int | type[int], ...] | None: Inferred shape tuple if found, else ``None``.
    """
    try:
        src_str = str(inspect.getsource(first))
        lines = src_str.splitlines()
        spaces = (len(lines[0]) - len(lines[0].lstrip())) if lines else 0
        src_code = chr(10).join(line[spaces:] if len(line) >= spaces else line for line in lines)
        return _extract_return_tuple_dims(src_code)
    except (OSError, TypeError, SyntaxError) as exc:
        logging.getLogger(__name__).debug("AST introspection failed for %s: %s", first, exc)
        return None


def _infer_shape_from_callable(
    first: ForeignCallable,
) -> tuple[int | type[int], ...] | None:
    """Infer output shape from a callable via annotations or AST introspection.

    Args:
        first (ForeignCallable): Target callable to inspect.

    Returns:
        tuple[int | type[int], ...] | None: Inferred shape tuple if found, else ``None``.
    """
    anno_shape = _infer_shape_from_callable_annotations(first)
    if anno_shape is not None:
        return anno_shape
    return _infer_shape_from_callable_ast(first)


@register_op("ForeignCall")
class ForeignCall(OpDef):
    """Universal ForeignCall operation definition for external callables and subgraphs."""

    def infer_shape(self, *args: ForeignTarget, **kwargs: IRAttrValue) -> tuple[int | type[int], ...]:
        """Infer the output shape for ForeignCall via attribute, annotation, or AST introspection.

        Args:
            *args (ForeignTarget): Positional arguments ``(callable_or_module, inputs, ...)``.
            **kwargs (IRAttrValue): Keyword attributes including optional ``output_shape``.

        Returns:
            tuple[int | type[int], ...]: Inferred output shape tuple.
        """
        if "output_shape" in kwargs and kwargs["output_shape"] is not None:
            shape_val = kwargs["output_shape"]
            if isinstance(shape_val, (list, tuple)):
                return tuple(_coerce_foreign_dim(d) for d in shape_val)
            return (_coerce_foreign_dim(shape_val),)

        if args:
            first = args[0]
            attr_shape = _infer_shape_from_attributes(first)
            if attr_shape is not None:
                return attr_shape

            if isinstance(
                first, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type, _ForeignCallableProtocol)
            ):
                call_shape = _infer_shape_from_callable(first)
                if call_shape is not None:
                    return call_shape

            if len(args) > 1 and isinstance(args[1], ShapeProtocol):
                return _extract_foreign_shape(extract_shape(args[1]))

        return ()


__all__ = [
    "ForeignCall",
    "OutputShapeHolderProtocol",
]
