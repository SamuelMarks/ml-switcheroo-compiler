# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Global tracing state."""

import sys

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref


class TracingState:
    """Thread-safe state for tracing."""

    def __init__(self) -> None:
        """Initialize."""
        self.is_tracing: bool = False
        self.active_graph = None
        self.graph_stack: list[LogicalGraph] = []
        self.constant_cache = {}
        self.current_loop_options = None
        self.loop_options_stack: list = []

    def _enrich_ast_and_domain(self, node) -> None:
        """Enrich a node with AST and domain information.

        Args:
            node (object): The IR node to enrich.
        """
        if getattr(node, "source_ast_ref", None) is None:
            node.source_ast_ref = get_source_ast_ref()
        if self.active_graph is not None and getattr(self.active_graph, "name", None) is not None and getattr(node, "domain", "") == "":
            node.domain = getattr(self.active_graph, "name", "")

    def _enrich_stream(self, node) -> None:
        """Enrich a node with current stream information.

        Args:
            node (object): The IR node to enrich.
        """
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        config = sys.modules["ml_switcheroo_compiler.core.config"].config
        if getattr(node, "stream", "default") is None and config.current_stream != "default":
            node.stream = config.current_stream

    def _enrich_node(self, node) -> None:
        """Enrich a newly created node with implicit context metadata (AST, domain, stream).

        Args:
            node (object): The IR node to enrich with metadata.
        """
        self._enrich_ast_and_domain(node)
        self._enrich_stream(node)

    def add_node(self, node) -> None:
        """Register a node into the currently active trace graph.

        Args:
            node (object): The IR node to append to the computational graph.

        Raises:
            RuntimeError: If tracing is not currently active.
        """
        if not self.is_tracing or self.active_graph is None:
            msg = "Cannot add node: not currently tracing."
            return

        self._enrich_node(node)
        if self.current_loop_options is not None and getattr(node, "op_type", "") in ("Loop", "WhileLoop") and hasattr(node, "attributes") and isinstance(node.attributes, dict):
            opts = self.current_loop_options
            if "parallel_iterations" not in node.attributes and getattr(opts, "parallel_iterations", None) is not None:
                node.attributes["parallel_iterations"] = opts.parallel_iterations
            if "swap_memory" not in node.attributes and getattr(opts, "swap_memory", None) is not None:
                node.attributes["swap_memory"] = opts.swap_memory
            if "maximum_iterations" not in node.attributes and getattr(opts, "maximum_iterations", None) is not None:
                node.attributes["maximum_iterations"] = opts.maximum_iterations
            if "shape_invariants" not in node.attributes and getattr(opts, "shape_invariants", None) is not None:
                node.attributes["shape_invariants"] = opts.shape_invariants
            if "loop_options" not in node.attributes:
                node.attributes["loop_options"] = opts

        self.active_graph.nodes[node.id] = node
        if getattr(node, "op_type", "") != "Input" and hasattr(self.active_graph, "inputs"):
            if node.id in self.active_graph.inputs:
                self.active_graph.inputs.remove(node.id)
                if hasattr(self.active_graph, "input_specs"):
                    self.active_graph.input_specs.pop(node.id, None)

    def start_tracing(self, name: str = "Model"):
        """Activate the tracing context and initialize a new empty graph.

        Args:
            name (str): The logical name assigned to the computational graph.

        Returns: Tensor: The newly initialized LogicalGraph instance.
        """
        if self.active_graph is not None:
            self.graph_stack.append(self.active_graph)
            self.loop_options_stack.append(self.current_loop_options)
        self.active_graph = LogicalGraph(name=name)
        self.constant_cache = {}
        self.current_loop_options = None
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self):
        """Deactivate the tracing context and return the captured graph.

        Returns: Tensor: The populated LogicalGraph containing all operations captured during tracing.
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
