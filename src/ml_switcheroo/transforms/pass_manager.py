"""Pass Manager Infrastructure for Middle-End Transformations."""

from typing import Callable
import hashlib
import json

from ml_switcheroo.ir.core import IRGraph, IRNode
from ml_switcheroo.core.errors import CompilationError


class DAGTopologicalSorter:
    """Topological Sorter for IR graphs."""

    @staticmethod
    def sort(graph: IRGraph) -> list[IRNode]:
        """Perform topological sort on the graph nodes."""
        visited: set[str] = set()
        temp_mark: set[str] = set()
        sorted_nodes: list[IRNode] = []

        def visit(node_id: str) -> None:
            """Docstring."""
            if node_id in temp_mark:
                raise CompilationError("Cycle detected in graph.")
            if node_id not in visited:
                temp_mark.add(node_id)
                node = graph.nodes.get(node_id)
                if node is not None:
                    for in_id in node.inputs:
                        visit(in_id)
                temp_mark.remove(node_id)
                visited.add(node_id)
                if node is not None:
                    sorted_nodes.append(node)

        # Visit all nodes
        for node_id in graph.nodes.keys():
            if node_id not in visited:
                visit(node_id)

        return sorted_nodes


class IRValidator:
    """Validator for checking IR graph structural integrity."""

    @staticmethod
    def check_cycles(graph: IRGraph) -> None:
        """Validate that the graph has no cycles."""
        DAGTopologicalSorter.sort(graph)

    @staticmethod
    def check_shapes(graph: IRGraph) -> None:
        """Validate shape consistency."""
        # Simple validator: ensure every node has shape metadata
        for node_id, node in graph.nodes.items():
            if getattr(node, "shape_metadata", None) is None:
                raise CompilationError(f"Node {node_id} is missing shape_metadata.")


def _graph_hash(graph: IRGraph) -> str:
    """Compute a deterministic hash of the graph structure."""
    state = {}
    for node_id, node in graph.nodes.items():
        state[node_id] = {
            "op": node.op_type,
            "inputs": node.inputs,
            # Ignore attributes that might not be easily serializable for basic hash
        }
    return hashlib.md5(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


class PassManager:
    """Manages and executes passes on an IR graph."""

    def __init__(self) -> None:
        """Initialize the PassManager."""
        self.passes: list[Callable[[IRGraph], bool]] = []
        self.validators: list[Callable[[IRGraph], None]] = [
            IRValidator.check_cycles,
            IRValidator.check_shapes,
        ]

    def add_pass(self, ir_pass: Callable[[IRGraph], bool]) -> None:
        """Add a pass to the manager.

        A pass should return True if it modified the graph.
        """
        self.passes.append(ir_pass)

    def validate(self, graph: IRGraph) -> None:
        """Run all validators on the graph."""
        for validator in self.validators:
            validator(graph)

    def run(self, graph: IRGraph) -> IRGraph:
        """Run all passes sequentially on the graph."""
        self.validate(graph)
        for ir_pass in self.passes:
            ir_pass(graph)
        self.validate(graph)
        return graph

    def run_until_converged(self, graph: IRGraph, max_iterations: int = 10) -> IRGraph:
        """Run passes until the graph stops changing or max_iterations reached."""
        self.validate(graph)

        for _ in range(max_iterations):
            prev_hash = _graph_hash(graph)

            for ir_pass in self.passes:
                ir_pass(graph)

            new_hash = _graph_hash(graph)
            if new_hash == prev_hash:
                break

        self.validate(graph)
        return graph
