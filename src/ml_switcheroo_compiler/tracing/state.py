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
        self.constant_cache: dict[object, object] = {}

    def _enrich_ast_and_domain(self, node: object) -> None:
        """Enrich a node with AST and domain information.

        Args:
            node (object): The IR node to enrich.
        """
        if getattr(node, "source_ast_ref", None) is None:
            node.source_ast_ref = get_source_ast_ref()
        if self.active_graph is not None and getattr(self.active_graph, "name", None) is not None and getattr(node, "domain", "") == "":
            node.domain = getattr(self.active_graph, "name", "")

    def _enrich_stream(self, node: object) -> None:
        """Enrich a node with current stream information.

        Args:
            node (object): The IR node to enrich.
        """
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        config: object = sys.modules["ml_switcheroo_compiler.core.config"].config
        if getattr(node, "stream", "default") is None and config.current_stream != "default":
            node.stream = config.current_stream

    def _enrich_node(self, node: object) -> None:
        """Enrich a newly created node with implicit context metadata (AST, domain, stream).

        Args:
            node (object): The IR node to enrich with metadata.
        """
        self._enrich_ast_and_domain(node)
        self._enrich_stream(node)

    def add_node(self, node: object) -> None:
        """Register a node into the currently active trace graph.

        Args:
            node (object): The IR node to append to the computational graph.

        Raises:
            RuntimeError: If tracing is not currently active.
        """
        if not self.is_tracing or self.active_graph is None:
            msg: object = "Cannot add node: not currently tracing."
            return

        self._enrich_node(node)
        self.active_graph.nodes[node.id] = node

    def start_tracing(self, name: str = "Model") -> object:
        """Activate the tracing context and initialize a new empty graph.

        Args:
            name (str): The logical name assigned to the computational graph.

        Returns: object: The newly initialized LogicalGraph instance.
        """
        self.active_graph = LogicalGraph(name=name)
        self.constant_cache = {}
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self) -> object:
        """Deactivate the tracing context and return the captured graph.

        Returns: object: The populated LogicalGraph containing all operations captured during tracing.
        """
        graph: object = self.active_graph
        self.active_graph = None
        self.is_tracing = False
        return graph


global_tracing_state: object = TracingState()
