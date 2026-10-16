"""Global tracing state."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

if TYPE_CHECKING:
    from ml_switcheroo_compiler.tracing.autograph import LoopOptions


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
        if getattr(node, "source_ast_ref", None) is None:
            node.source_ast_ref = get_source_ast_ref()
        if self.active_graph is not None and getattr(self.active_graph, "name", None) is not None and getattr(node, "domain", "") == "":
            node.domain = getattr(self.active_graph, "name", "")

    def _enrich_stream(self, node: LogicalNode) -> None:
        """Enrich a node with current stream information.

        Args:
            node (LogicalNode): The IR node to enrich.
        """
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        config = sys.modules["ml_switcheroo_compiler.core.config"].config
        if getattr(node, "stream", "default") is None and config.current_stream != "default":
            node.stream = config.current_stream

    def _enrich_node(self, node: LogicalNode) -> None:
        """Enrich a newly created node with implicit context metadata (AST, domain, stream).

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
        attrs = getattr(node, "attributes", None)
        if self.current_loop_options is None or getattr(node, "op_type", "") not in ("Loop", "WhileLoop") or not isinstance(attrs, dict):
            return
        opts = self.current_loop_options
        for key in ("parallel_iterations", "swap_memory", "maximum_iterations", "shape_invariants"):
            val = getattr(opts, key, None)
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
        if getattr(node, "op_type", "") != "Input" and hasattr(self.active_graph, "inputs"):
            if node.id in self.active_graph.inputs:
                self.active_graph.inputs.remove(node.id)
                if hasattr(self.active_graph, "input_specs"):
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
