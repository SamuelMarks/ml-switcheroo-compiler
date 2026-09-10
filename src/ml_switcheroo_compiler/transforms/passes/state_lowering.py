# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module state_lowering.py."""

"""State Lowering pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def state_lowering_pass(graph: IRGraph) -> bool:
    """In-place State Lowering pass.

    Lowers functional I/O bounds into explicit state assignments (e.g. ReadVariable/AssignVariable).

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
    """Reverse lowering pass emitting stateful object-oriented class definitions from purely functional graphs."""

    def __init__(self, target_framework: str = "pytorch") -> None:
        """Initialize StateLoweringPass.

        Args:
            target_framework (str): Target framework name ("pytorch", "jax", "keras").
        """
        self.target_framework = target_framework.lower()

    def run(self, graph: IRGraph) -> bool:
        """Execute state lowering transformation on graph.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            bool: True if graph was modified.
        """
        return state_lowering_pass(graph)

    def emit_stateful_class(
        self,
        graph: IRGraph,
        class_name: str = "LoweredModule",
    ) -> str:
        """Emit a stateful object-oriented class definition from a functional graph.

        Args:
            graph (IRGraph): Functional IR graph with parameter inputs and outputs.
            class_name (str): Name of the generated stateful class.

        Returns:
            str: Generated Python source code representing the stateful class.
        """
        state_inputs = [n for n in graph.nodes.values() if (n.op_type == "Input" and n.attributes.get("is_state", False)) or n.op_type == "ReadVariable"]
        non_state_inputs = [n for n in graph.nodes.values() if (n.op_type == "Input" and not n.attributes.get("is_state", False))]

        fw = self.target_framework
        lines = []

        if fw in ("pytorch", "torch"):
            lines.append("import torch")
            lines.append("import torch.nn as nn\n")
            lines.append(f"class {class_name}(nn.Module):")
            lines.append('    """Generated stateful module from functional IRGraph."""\n')
            lines.append("    def __init__(self) -> None:")
            lines.append("        super().__init__()")
            if not state_inputs:
                lines.append("        pass")
            for node in state_inputs:
                param_name = node.attributes.get("param_name", node.attributes.get("variable_name", node.id))
                clean_name = str(param_name).replace(".", "_")
                shape = node.shape_metadata.get("shape", (1, 1)) if hasattr(node, "shape_metadata") and node.shape_metadata else (1, 1)
                if node.attributes.get("is_buffer", False):
                    lines.append(f"        self.register_buffer('{clean_name}', torch.zeros({shape}))")
                else:
                    lines.append(f"        self.{clean_name} = nn.Parameter(torch.zeros({shape}))")

            args_str = ", ".join(["self"] + [inp.id for inp in non_state_inputs])
            lines.append(f"\n    def forward({args_str}):")
            if not graph.outputs:
                lines.append("        return None")
            else:
                lines.append(f"        return {', '.join(graph.outputs)}")

        elif fw == "jax":
            lines.append("import jax")
            lines.append("import jax.numpy as jnp\n")
            lines.append(f"class {class_name}:")
            lines.append('    """Generated stateful module from functional IRGraph."""\n')
            lines.append("    def __init__(self) -> None:")
            if not state_inputs:
                lines.append("        pass")
            for node in state_inputs:
                param_name = node.attributes.get("param_name", node.attributes.get("variable_name", node.id))
                clean_name = str(param_name).replace(".", "_")
                shape = node.shape_metadata.get("shape", (1, 1)) if hasattr(node, "shape_metadata") and node.shape_metadata else (1, 1)
                lines.append(f"        self.{clean_name} = jnp.zeros({shape})")

            args_str = ", ".join(["self"] + [inp.id for inp in non_state_inputs])
            lines.append(f"\n    def __call__({args_str}):")
            if not graph.outputs:
                lines.append("        return None")
            else:
                lines.append(f"        return {', '.join(graph.outputs)}")

        else:
            lines.append(f"class {class_name}:")
            lines.append('    """Generated stateful module from functional IRGraph."""\n')
            lines.append("    def __init__(self) -> None:")
            lines.append("        pass")
            args_str = ", ".join(["self"] + [inp.id for inp in non_state_inputs])
            lines.append(f"\n    def __call__({args_str}):")
            lines.append("        return None")

        return "\n".join(lines) + "\n"
