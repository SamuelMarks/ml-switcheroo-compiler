"""State Lowering pass."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def state_lowering_pass(graph: IRGraph) -> bool:
    """In-place State Lowering pass.

    Lowers functional I/O bounds into explicit state assignments (`ReadVariable` / `AssignVariable`).

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Input" and node.attributes.get("is_state", False):
            node.op_type = "ReadVariable"
            node.attributes["variable_name"] = node.attributes.get("name", node.id)
            if "name" in node.attributes:
                del node.attributes["name"]
            modified = True
        elif node.op_type == "Output" and node.attributes.get("is_state", False):
            node.op_type = "AssignVariable"
            node.attributes["variable_name"] = node.attributes.get("name", node.id)
            if "name" in node.attributes:
                del node.attributes["name"]
            modified = True

    return modified


class StateLoweringPass:
    """Symbolic lowering pass delegating stateful class codegen to the `backends` subsystem."""

    def __init__(self, target_framework: str = "pytorch") -> None:
        """Initialize StateLoweringPass.

        Args:
            target_framework (str): Target backend emitter identifier.
        """
        self.target_framework = target_framework.lower()

    def run(self, graph: IRGraph) -> bool:
        """Execute symbolic state lowering transformation on `graph`.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            bool: True if `graph` was modified.
        """
        return state_lowering_pass(graph)

    def emit_stateful_class(
        self,
        graph: IRGraph,
        class_name: str = "LoweredModule",
    ) -> str:
        """Delegate stateful class source generation to `backends.stateful_emitters`.

        Args:
            graph (IRGraph): Functional IR graph with parameter inputs and outputs.
            class_name (str): Name of the generated stateful class.

        Returns:
            str: Generated source code representing the stateful class.
        """
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        return BackendRegistry.emit_stateful_class(self.target_framework, graph, class_name=class_name)
