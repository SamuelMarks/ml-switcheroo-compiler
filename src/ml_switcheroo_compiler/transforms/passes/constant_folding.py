"""Constant Folding pass evaluated via reference eager backend."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import AttributeValue, IRGraph, IRNode, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.interpreter.evaluator import InterpreterValue
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


@typing.runtime_checkable
class _SizeOneProtocol(typing.Protocol):
    """Protocol for tensor/array values exposing a `size` attribute."""

    size: int


@typing.runtime_checkable
class _NumelProtocol(typing.Protocol):
    """Protocol for tensor values exposing a `.numel()` method."""

    def numel(self) -> int:
        """Return total element count.

        Returns:
            int: Element count.
        """
        ...


@typing.runtime_checkable
class _BackendItemProtocol(typing.Protocol):
    """Protocol for backend classes/modules exposing `.item(val)`."""

    def item(self, val: InterpreterValue) -> InterpreterValue:
        """Extract scalar item from a 1-element array.

        Args:
            val (InterpreterValue): Array value.

        Returns:
            InterpreterValue: Extracted scalar value.
        """
        ...


@typing.runtime_checkable
class _ScalarItemProtocol(typing.Protocol):
    """Protocol for array values exposing `.item()`."""

    def item(self) -> InterpreterValue:
        """Extract scalar item.

        Returns:
            InterpreterValue: Extracted scalar value.
        """
        ...


def _has_size_attr(val: InterpreterValue) -> typing.TypeGuard[_SizeOneProtocol]:
    """Check whether `val` exposes a `size` attribute.

    Args:
        val (InterpreterValue): Evaluated array or scalar.

    Returns:
        bool: True if `val` satisfies `_SizeOneProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _SizeOneProtocol)
    return found or is_mock_with_attrs(val, "size")


def _has_numel_method(val: InterpreterValue) -> typing.TypeGuard[_NumelProtocol]:
    """Check whether `val` exposes a `numel` method.

    Args:
        val (InterpreterValue): Evaluated array or scalar.

    Returns:
        bool: True if `val` satisfies `_NumelProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _NumelProtocol)
    return found or is_mock_with_attrs(val, "numel")


def _has_backend_item(val: type | OpValue) -> typing.TypeGuard[_BackendItemProtocol]:
    """Check whether `val` exposes an `item` method.

    Args:
        val (type | OpValue): Backend class or instance.

    Returns:
        bool: True if `val` satisfies `_BackendItemProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _BackendItemProtocol)
    return found or is_mock_with_attrs(val, "item")


def _has_scalar_item(val: InterpreterValue) -> typing.TypeGuard[_ScalarItemProtocol]:
    """Check whether `val` exposes an `item` method.

    Args:
        val (InterpreterValue): Evaluated array.

    Returns:
        bool: True if `val` satisfies `_ScalarItemProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ScalarItemProtocol)
    return found or is_mock_with_attrs(val, "item")


def _are_all_inputs_constant(canonical_inputs: list[str], graph: IRGraph) -> bool:
    """Check if all inputs to a node are compile-time known constants.

    Args:
        canonical_inputs (list[str]): List of canonical input IDs.
        graph (IRGraph): Computation graph containing nodes.

    Returns:
        bool: True if every input is an existing Constant node.
    """
    if not canonical_inputs:
        return False
    for inp in canonical_inputs:
        if inp not in graph.nodes or graph.nodes[inp].op_type != "Constant":
            return False
    return True


def _evaluate_constant_node(
    node: IRNode,
    canonical_inputs: list[str],
    graph: IRGraph,
    backend: type | OpValue,
) -> InterpreterValue:
    """Evaluate a constant node using the reference eager backend.

    Args:
        node (IRNode): Primal IR node to fold.
        canonical_inputs (list[str]): Input node IDs.
        graph (IRGraph): Parent computation graph.
        backend (type | OpValue): Reference eager backend generator or instance.

    Returns:
        InterpreterValue: Evaluated concrete value.
    """
    subgraph = LogicalGraph(outputs=[node.id])
    for inp in canonical_inputs:
        subgraph.nodes[inp] = graph.nodes[inp]
    subgraph.nodes[node.id] = LogicalNode(
        id=node.id,
        op_type=node.op_type,
        attributes=dict(node.attributes),
        inputs=list(canonical_inputs),
        shape_metadata=node.shape_metadata,
    )
    outputs = evaluate_graph(subgraph, {}, backend=backend)
    val: InterpreterValue = outputs[node.id]
    raw_val: InterpreterValue = val
    is_scalar_one = (_has_size_attr(raw_val) and raw_val.size == 1) or (_has_numel_method(val) and val.numel() == 1)
    if is_scalar_one:
        raw_out: InterpreterValue = outputs[node.id]
        if _has_backend_item(backend):
            val = backend.item(raw_out)
        elif _has_scalar_item(raw_out):
            val = raw_out.item()
    return val


def _resolve_folding_backend() -> type | None:
    """Resolve the active or fallback reference backend for constant evaluation.

    Returns:
        type | None: Backend class or None if unavailable.
    """
    backend: type | None = get_active_backend()
    if backend is None:
        try:
            backend = BackendRegistry.get("pure_python")
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ):
            backend = None
    return backend


_NON_FOLDABLE_CALLBACK_OPS: frozenset[str] = frozenset(
    {
        "Constant",
        "ProcessCustomVJPCall",
        "ProcessCustomJVPCall",
    }
)


def _try_fold_single_node(
    node: IRNode,
    canonical_inputs: list[str],
    graph: IRGraph,
    backend: type | None,
) -> bool:
    """Attempt to fold a single node with constant inputs into a Constant node.

    Args:
        node (IRNode): Node to evaluate and fold.
        canonical_inputs (list[str]): Resolved input node IDs.
        graph (IRGraph): Containing computation graph.
        backend (type | None): Reference eager backend.

    Returns:
        bool: True if the node was folded into a Constant, False otherwise.
    """
    import logging

    if node.op_type in _NON_FOLDABLE_CALLBACK_OPS or not _are_all_inputs_constant(canonical_inputs, graph):
        return False

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    try:
        if backend is None:
            return False

        from ml_switcheroo_compiler.ir.core import ensure_attr_val

        val = _evaluate_constant_node(node, canonical_inputs, graph, backend)
        graph.nodes[node.id].op_type = "Constant"
        new_attrs: dict[str, AttributeValue] = {"value": ensure_attr_val(val)}
        graph.nodes[node.id].attributes = new_attrs
        graph.nodes[node.id].inputs = []
        return True
    except UnimplementedMathError as e:
        logging.getLogger(__name__).debug(f"Unimplemented math for constant folding {node.op_type}: {e}")
        return False
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as e:
        logging.getLogger(__name__).debug(f"Failed to constant fold {node.op_type}: {e}")
        return False


def _fold_nested_subgraphs(graph: IRGraph) -> bool:
    """Recursively run constant folding on nested subgraphs in nodes and attributes.

    Args:
        graph (IRGraph): Parent computation graph.

    Returns:
        bool: True if any nested subgraph was modified, False otherwise.
    """
    modified = False
    for node in graph.nodes.values():
        for sub in node.subgraphs.values():
            if isinstance(sub, LogicalGraph) and constant_folding_pass(sub):
                modified = True
        for raw_attr_val in node.attributes.values():
            attr_val: LogicalGraph | AttributeValue = raw_attr_val
            if isinstance(attr_val, LogicalGraph) and constant_folding_pass(attr_val):
                modified = True
    return modified


def constant_folding_pass(graph: IRGraph) -> bool:
    """In-place Constant Folding evaluated via reference eager backend.

    Evaluates compile-time known constant nodes until fixpoint is reached.

    Args:
        graph (IRGraph): Input `graph` argument (graph).

    Returns:
        bool: True if any nodes were folded into constants.
    """
    total_modified = False
    pass_modified = True

    while pass_modified:
        pass_modified = False
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        id_map: dict[str, str] = {}
        backend = _resolve_folding_backend()

        for node in sorted_nodes:
            canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]
            if _try_fold_single_node(node, canonical_inputs, graph, backend):
                pass_modified = True
                total_modified = True
            id_map[node.id] = node.id

    if _fold_nested_subgraphs(graph):
        total_modified = True

    return total_modified
