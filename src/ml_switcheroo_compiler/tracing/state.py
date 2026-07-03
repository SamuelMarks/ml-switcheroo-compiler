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

    def add_node(self, node: object) -> None:
        """Add node.

        Args:
            node (object): node
        """
        if not self.is_tracing or self.active_graph is None:
            msg = "Cannot add node: not currently tracing."
            raise RuntimeError(msg)

        if getattr(node, "source_ast_ref", None) is None:
            node.source_ast_ref = get_source_ast_ref()

        if self.active_graph.name is not None and getattr(node, "domain", "") == "":
            node.domain = self.active_graph.name

        if "ml_switcheroo_compiler.core.config" in sys.modules:
            config = sys.modules["ml_switcheroo_compiler.core.config"].config
            if hasattr(node, "stream") and node.stream is None and config.current_stream != "default":
                node.stream = config.current_stream

        self.active_graph.nodes[node.id] = node

    def start_tracing(self, name: str = "Model") -> object:
        """Start tracing.

        Args:
            name (str): name

        Returns:
            object: graph
        """
        self.active_graph = LogicalGraph(name=name)
        self.constant_cache = {}
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self) -> object:
        """Stop tracing.

        Returns:
            object: graph
        """
        graph = self.active_graph
        self.active_graph = None
        self.is_tracing = False
        return graph


global_tracing_state = TracingState()
