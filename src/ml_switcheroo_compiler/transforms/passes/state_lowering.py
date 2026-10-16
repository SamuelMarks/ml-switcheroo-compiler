"""State Lowering pass."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
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


def _node_clean_name_and_shape(node: IRNode) -> tuple[str, tuple[int, ...]]:
    """Extract sanitized attribute name and default shape tuple from a state node.

    Args:
        node (IRNode): State input or ReadVariable node.

    Returns:
        tuple[str, tuple[int, ...]]: Sanitized attribute name and shape tuple.
    """
    param_name = node.attributes.get("param_name", node.attributes.get("variable_name", node.id))
    clean_name = str(param_name).replace(".", "_")
    shape_meta = getattr(node, "shape_metadata", None)
    shape = shape_meta.get("shape", (1, 1)) if isinstance(shape_meta, dict) else (1, 1)
    return clean_name, shape


def _emit_pytorch_class(
    class_name: str,
    state_inputs: list[IRNode],
    args_str: str,
    return_stmt: str,
) -> list[str]:
    """Emit lines for a PyTorch nn.Module class definition.

    Args:
        class_name (str): Name of the generated class.
        state_inputs (list[IRNode]): State parameter and buffer nodes.
        args_str (str): Formatted method parameter list string.
        return_stmt (str): Formatted return statement line.

    Returns:
        list[str]: Source code lines for the PyTorch module.
    """
    lines = [
        "import torch",
        "import torch.nn as nn\n",
        f"class {class_name}(nn.Module):",
        '    """Generated stateful module from functional IRGraph."""\n',
        "    def __init__(self) -> None:",
        "        super().__init__()",
    ]
    if not state_inputs:
        lines.append("        pass")
    for node in state_inputs:
        clean_name, shape = _node_clean_name_and_shape(node)
        if node.attributes.get("is_buffer", False):
            lines.append(f"        self.register_buffer('{clean_name}', torch.zeros({shape}))")
        else:
            lines.append(f"        self.{clean_name} = nn.Parameter(torch.zeros({shape}))")
    lines.append(f"\n    def forward({args_str}):")
    lines.append(return_stmt)
    return lines


def _emit_jax_class(
    class_name: str,
    state_inputs: list[IRNode],
    args_str: str,
    return_stmt: str,
) -> list[str]:
    """Emit lines for a JAX stateful class definition.

    Args:
        class_name (str): Name of the generated class.
        state_inputs (list[IRNode]): State parameter and buffer nodes.
        args_str (str): Formatted method parameter list string.
        return_stmt (str): Formatted return statement line.

    Returns:
        list[str]: Source code lines for the JAX class.
    """
    lines = [
        "import jax",
        "import jax.numpy as jnp\n",
        f"class {class_name}:",
        '    """Generated stateful module from functional IRGraph."""\n',
        "    def __init__(self) -> None:",
    ]
    if not state_inputs:
        lines.append("        pass")
    for node in state_inputs:
        clean_name, shape = _node_clean_name_and_shape(node)
        lines.append(f"        self.{clean_name} = jnp.zeros({shape})")
    lines.append(f"\n    def __call__({args_str}):")
    lines.append(return_stmt)
    return lines


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
        non_state_inputs = [n for n in graph.nodes.values() if n.op_type == "Input" and not n.attributes.get("is_state", False)]

        args_str = ", ".join(["self"] + [inp.id for inp in non_state_inputs])
        return_stmt = f"        return {', '.join(graph.outputs)}" if graph.outputs else "        return None"

        if self.target_framework in ("pytorch", "torch"):
            lines = _emit_pytorch_class(class_name, state_inputs, args_str, return_stmt)
        elif self.target_framework == "jax":
            lines = _emit_jax_class(class_name, state_inputs, args_str, return_stmt)
        else:
            lines = [
                f"class {class_name}:",
                '    """Generated stateful module from functional IRGraph."""\n',
                "    def __init__(self) -> None:",
                "        pass",
                f"\n    def __call__({args_str}):",
                "        return None",
            ]

        return "\n".join(lines) + "\n"
