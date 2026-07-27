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
        self.constant_cache: dict[object, object] = {}

    def _enrich_ast_and_domain(self, node: object) -> None:
        if getattr(node, "source_ast_ref", None) is None:
            node.source_ast_ref = get_source_ast_ref()
        if self.active_graph.name is not None and getattr(node, "domain", "") == "":
            node.domain = self.active_graph.name

    def _enrich_stream(self, node: object) -> None:
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        config = sys.modules["ml_switcheroo_compiler.core.config"].config
        if getattr(node, "stream", "default") is None and config.current_stream != "default":
            node.stream = config.current_stream

    def _enrich_node(self, node: object) -> None:
        """Enrich node with implicit metadata."""
        self._enrich_ast_and_domain(node)
        self._enrich_stream(node)

    def add_node(self, node: object) -> None:
        """Add node.

        Args:
            node (object): node
        """
        if not self.is_tracing or self.active_graph is None:
            msg = "Cannot add node: not currently tracing."
            raise RuntimeError(msg)

        self._enrich_node(node)
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
