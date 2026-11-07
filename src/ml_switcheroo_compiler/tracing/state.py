"""Global tracing state."""

from __future__ import annotations

import sys
import typing

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref
from ml_switcheroo_compiler.backends.protocols import has_backend_member

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from ml_switcheroo_ir import AttributeValue
    from ml_switcheroo_ir.types import TensorSpec

    from ml_switcheroo_compiler.core.shape import TensorOrScalar
    from ml_switcheroo_compiler.tracing.autograph import LoopOptions, ShapeInvariantType


class _GraphInputsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `inputs`."""

    inputs: list[str]


class _GraphInputSpecsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `input_specs`."""

    input_specs: dict[str, TensorSpec]


def _is_attr_dict(val: Mapping[str, AttributeValue]) -> typing.TypeGuard[dict[str, LoopOptions | TensorOrScalar | AttributeValue]]:
    """Type guard narrowing node attributes dict to accept `LoopOptions`.

    Args:
        val (Mapping[str, AttributeValue]): Node attributes dictionary.

    Returns:
        bool: Always True for runtime dicts.
    """
    return isinstance(val, dict)


class _NodeAstRefProtocol(typing.Protocol):
    """Protocol for nodes exposing `source_ast_ref`."""

    source_ast_ref: str | None


class _NodeDomainProtocol(typing.Protocol):
    """Protocol for nodes exposing `domain`."""

    domain: str


class _NodeStreamProtocol(typing.Protocol):
    """Protocol for nodes exposing `stream`."""

    stream: str | None


class _NodeLoopAttrsProtocol(typing.Protocol):
    """Protocol for loop nodes exposing `op_type` and `attributes`."""

    op_type: str
    attributes: dict[str, AttributeValue]


class _NodeOpTypeProtocol(typing.Protocol):
    """Protocol for nodes exposing `id` and `op_type`."""

    id: str
    op_type: str


class _GraphNameProtocol(typing.Protocol):
    """Protocol for graphs exposing `name`."""

    name: str | None


def _has_graph_inputs(val: LogicalGraph) -> typing.TypeGuard[_GraphInputsProtocol]:
    """Check whether `val` exposes `inputs`.

    Args:
        val (LogicalGraph): Active graph.

    Returns:
        bool: True if `val` has `inputs`.
    """
    return has_backend_member(val, "inputs")


def _has_graph_input_specs(val: LogicalGraph | _GraphInputsProtocol) -> typing.TypeGuard[_GraphInputSpecsProtocol]:
    """Check whether `val` exposes `input_specs`.

    Args:
        val (LogicalGraph | _GraphInputsProtocol): Active graph.

    Returns:
        bool: True if `val` has `input_specs`.
    """
    return has_backend_member(val, "input_specs")


def _has_node_ast_ref(val: LogicalNode) -> typing.TypeGuard[_NodeAstRefProtocol]:
    """Check whether `val` exposes `source_ast_ref`.

    Args:
        val (LogicalNode): Target node.

    Returns:
        bool: True if `val` has `source_ast_ref`.
    """
    return has_backend_member(val, "source_ast_ref")


def _has_node_domain(val: LogicalNode) -> typing.TypeGuard[_NodeDomainProtocol]:
    """Check whether `val` exposes `domain`.

    Args:
        val (LogicalNode): Target node.

    Returns:
        bool: True if `val` has `domain`.
    """
    return has_backend_member(val, "domain")


def _has_node_stream(val: LogicalNode) -> typing.TypeGuard[_NodeStreamProtocol]:
    """Check whether `val` exposes `stream`.

    Args:
        val (LogicalNode): Target node.

    Returns:
        bool: True if `val` has `stream`.
    """
    return has_backend_member(val, "stream")


def _has_node_loop_attrs(val: LogicalNode) -> typing.TypeGuard[_NodeLoopAttrsProtocol]:
    """Check whether `val` exposes `op_type` and `attributes`.

    Args:
        val (LogicalNode): Target node.

    Returns:
        bool: True if `val` has `op_type` and `attributes`.
    """
    return has_backend_member(val, "op_type") and has_backend_member(val, "attributes")


def _has_node_op_type(val: LogicalNode) -> typing.TypeGuard[_NodeOpTypeProtocol]:
    """Check whether `val` exposes `id` and `op_type`.

    Args:
        val (LogicalNode): Target node.

    Returns:
        bool: True if `val` has `id` and `op_type`.
    """
    return has_backend_member(val, "id") and has_backend_member(val, "op_type")


class TracingState:
    """Thread-safe state for tracing."""

    def __init__(self) -> None:
        """Initialize."""
        self.is_tracing: bool = False
        self.active_graph: LogicalGraph | None = None
        self.graph_stack: list[LogicalGraph] = []
        self.constant_cache: dict[int, str] = {}
        self.current_loop_options: LoopOptions | None = None
        self.loop_options_stack: list[LoopOptions | None] = []

    def _enrich_ast_and_domain(self, node: LogicalNode) -> None:
        """Enrich a node with AST and domain information.

        Args:
            node (LogicalNode): The IR node to enrich.
        """
        if _has_node_ast_ref(node) and node.source_ast_ref is None:
            node.source_ast_ref = get_source_ast_ref()
        active_g = self.active_graph
        if active_g is not None and active_g.name and _has_node_domain(node) and node.domain in (None, ""):
            node.domain = active_g.name

    def _enrich_stream(self, node: LogicalNode) -> None:
        """Enrich a node with current stream information.

        Args:
            node (LogicalNode): The IR node to enrich.
        """
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        from ml_switcheroo_compiler.core.config import config as active_config

        if _has_node_stream(node) and node.stream is None and active_config.current_stream != "default":
            node.stream = active_config.current_stream

    def _enrich_node(self, node: LogicalNode) -> None:
        """Enrich a newly created node with implicit context metadata (AST, domain, stream).

        Args:
            node (LogicalNode): The IR node to enrich with metadata.
        """
        self._enrich_node_fields(node)

    def _enrich_node_fields(self, node: LogicalNode) -> None:
        """Populate AST, domain, and stream fields on `node`.

        Args:
            node (LogicalNode): The IR node to enrich with metadata.
        """
        self._enrich_ast_and_domain(node)
        self._enrich_stream(node)

    def _apply_loop_options_to_node(self, node: LogicalNode) -> None:
        """Attach active loop options metadata to a Loop or WhileLoop node.

        Args:
            node (LogicalNode): The IR node to populate with loop options.
        """
        if self.current_loop_options is None or not _has_node_loop_attrs(node) or node.op_type not in ("Loop", "WhileLoop") or not isinstance(node.attributes, dict):
            return
        attrs = node.attributes
        assert _is_attr_dict(attrs)
        opts = self.current_loop_options
        opt_items: tuple[tuple[str, int | bool | ShapeInvariantType | None], ...] = (
            ("parallel_iterations", opts.parallel_iterations),
            ("swap_memory", opts.swap_memory),
            ("maximum_iterations", opts.maximum_iterations),
            ("shape_invariants", opts.shape_invariants),
        )
        for key, val in opt_items:
            if key not in attrs and val is not None:
                attrs[key] = val
        if "loop_options" not in attrs:
            attrs["loop_options"] = opts

    def add_node(self, node: LogicalNode) -> None:
        """Register a node into the currently active trace graph.

        Args:
            node (LogicalNode): The IR node to append to the computational graph.
        """
        if not self.is_tracing or self.active_graph is None:
            return

        self._enrich_node(node)
        self._apply_loop_options_to_node(node)

        self.active_graph.nodes[node.id] = node
        op_type = node.op_type if _has_node_op_type(node) else None
        if op_type != "Input" and _has_graph_inputs(self.active_graph):
            if node.id in self.active_graph.inputs:
                self.active_graph.inputs.remove(node.id)
                if _has_graph_input_specs(self.active_graph):
                    self.active_graph.input_specs.pop(node.id, None)

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Activate the tracing context and initialize a new empty graph.

        Args:
            name (str): The logical name assigned to the computational graph.

        Returns:
            LogicalGraph: The newly initialized LogicalGraph instance.
        """
        if self.active_graph is not None:
            self.graph_stack.append(self.active_graph)
            self.loop_options_stack.append(self.current_loop_options)
        self.active_graph = LogicalGraph(name=name)
        self.constant_cache = {}
        self.current_loop_options = None
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self) -> LogicalGraph | None:
        """Deactivate the tracing context and return the captured graph.

        Returns:
            LogicalGraph | None: The populated LogicalGraph containing all operations captured during tracing.
        """
        graph = self.active_graph
        if self.graph_stack:
            self.active_graph = self.graph_stack.pop()
            self.current_loop_options = self.loop_options_stack.pop() if self.loop_options_stack else None
            self.is_tracing = True
        else:
            self.active_graph = None
            self.current_loop_options = None
            self.loop_options_stack.clear()
            self.is_tracing = False
        return graph


global_tracing_state = TracingState()
