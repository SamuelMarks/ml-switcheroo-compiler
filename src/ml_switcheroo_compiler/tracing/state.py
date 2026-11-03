"""Global tracing state."""

from __future__ import annotations

import sys
import typing

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from ml_switcheroo_ir import AttributeValue
    from ml_switcheroo_ir.types import TensorSpec

    from ml_switcheroo_compiler.core.shape import StructuralOperandProtocol, TensorOrScalar
    from ml_switcheroo_compiler.tracing.autograph import LoopOptions


@typing.runtime_checkable
class _GraphInputsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `inputs`."""

    inputs: list[str]


@typing.runtime_checkable
class _GraphInputSpecsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `input_specs`."""

    input_specs: dict[str, TensorSpec]


def _is_attr_dict(val: Mapping[str, AttributeValue]) -> typing.TypeGuard[dict[str, StructuralOperandProtocol | TensorOrScalar]]:
    """Type guard narrowing node attributes dict to accept `StructuralOperandProtocol`.

    Args:
        val (Mapping[str, AttributeValue]): Node attributes dictionary.

    Returns:
        bool: Always True for runtime dicts.
    """
    return isinstance(val, dict)


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
        node_vars = vars(node)
        if node_vars.get("source_ast_ref") is None:
            node.source_ast_ref = get_source_ast_ref()
        active_g: StructuralOperandProtocol | None = self.active_graph
        if active_g is not None and self.active_graph is not None and vars(active_g).get("name") is not None and node_vars.get("domain", "") in (None, ""):
            node.domain = self.active_graph.name

    def _enrich_stream(self, node: LogicalNode) -> None:
        """Enrich a node with current stream information.

        Args:
            node (LogicalNode): The IR node to enrich.
        """
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        config = sys.modules["ml_switcheroo_compiler.core.config"].config
        if vars(node).get("stream", "default") is None and config.current_stream != "default":
            node.stream = config.current_stream

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
        from ml_switcheroo_compiler.ir.core import as_attrs

        node_vars = vars(node)
        attrs = node_vars.get("attributes")
        if self.current_loop_options is None or node_vars.get("op_type", "") not in ("Loop", "WhileLoop") or not isinstance(attrs, dict):
            return
        opts = self.current_loop_options
        raw_attrs: dict[str, TensorOrScalar] = {}
        opt_items: tuple[tuple[str, TensorOrScalar], ...] = (
            ("parallel_iterations", opts.parallel_iterations),
            ("swap_memory", opts.swap_memory),
            ("maximum_iterations", opts.maximum_iterations),
            ("shape_invariants", opts.shape_invariants),
        )
        for key, val in opt_items:
            if key not in attrs and val is not None:
                raw_attrs[key] = val
        attrs.update(as_attrs(raw_attrs))
        if "loop_options" not in attrs:
            opts_ref: StructuralOperandProtocol | TensorOrScalar = opts
            assert _is_attr_dict(attrs)
            attrs["loop_options"] = opts_ref

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
        if vars(node).get("op_type", "") != "Input" and isinstance(self.active_graph, _GraphInputsProtocol):
            if node.id in self.active_graph.inputs:
                self.active_graph.inputs.remove(node.id)
                if isinstance(self.active_graph, _GraphInputSpecsProtocol):
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
