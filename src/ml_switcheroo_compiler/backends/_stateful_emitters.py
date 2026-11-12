"""Backend-specific stateful class source emitters for lowered functional IR graphs."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import LogicalGraph as IRGraph
from ml_switcheroo_ir import LogicalNode as IRNode

from ml_switcheroo_compiler.core.shape import TensorOrScalar


@typing.runtime_checkable
class _ShapeMetaNodeProtocol(typing.Protocol):
    """Protocol for IR nodes exposing `shape_metadata`."""

    shape_metadata: dict[str, tuple[int, ...]] | tuple[int | str, ...] | list[int | str] | None


def _node_clean_name_and_shape(node: IRNode) -> tuple[str, tuple[int, ...]]:
    """Extract sanitized attribute name and default shape tuple from a state node.

    Args:
        node (IRNode): State input or ReadVariable node.

    Returns:
        tuple[str, tuple[int, ...]]: Sanitized attribute name and shape tuple.
    """
    param_name = node.attributes.get("param_name", node.attributes.get("variable_name", node.id))
    clean_name = str(param_name).replace(".", "_")

    raw_node: TensorOrScalar = node
    shape_meta = raw_node.shape_metadata if isinstance(raw_node, _ShapeMetaNodeProtocol) else None
    raw_shape = shape_meta.get("shape", (1, 1)) if isinstance(shape_meta, dict) else (1, 1)
    shape = tuple(int(x) for x in raw_shape)
    return clean_name, shape


def _emit_pytorch_class(
    class_name: str,
    state_inputs: list[IRNode],
    args_str: str,
    return_stmt: str,
) -> list[str]:
    """Emit lines for a PyTorch `nn.Module` class definition.

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
        "import torch.nn as nn",
        "",
        f"class {class_name}(nn.Module):",
        '    """Generated stateful module from functional IRGraph."""',
        "",
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
    lines.extend(["", f"    def forward({args_str}):", return_stmt])
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
        "import jax.numpy as jnp",
        "",
        f"class {class_name}:",
        '    """Generated stateful module from functional IRGraph."""',
        "",
        "    def __init__(self) -> None:",
    ]
    if not state_inputs:
        lines.append("        pass")
    for node in state_inputs:
        clean_name, shape = _node_clean_name_and_shape(node)
        lines.append(f"        self.{clean_name} = jnp.zeros({shape})")
    lines.extend(["", f"    def __call__({args_str}):", return_stmt])
    return lines


def _emit_keras_class(
    class_name: str,
    state_inputs: list[IRNode],
    args_str: str,
    return_stmt: str,
) -> list[str]:
    """Emit lines for a Keras `keras.layers.Layer` class definition.

    Args:
        class_name (str): Name of the generated class.
        state_inputs (list[IRNode]): State parameter and buffer nodes.
        args_str (str): Formatted method parameter list string.
        return_stmt (str): Formatted return statement line.

    Returns:
        list[str]: Source code lines for the Keras layer.
    """
    lines = [
        "import keras",
        "",
        f"class {class_name}(keras.layers.Layer):",
        '    """Generated stateful layer from functional IRGraph."""',
        "",
        "    def __init__(self) -> None:",
        "        super().__init__()",
    ]
    if not state_inputs:
        lines.append("        pass")
    for node in state_inputs:
        clean_name, shape = _node_clean_name_and_shape(node)
        trainable = not bool(node.attributes.get("is_buffer", False))
        lines.append(f"        self.{clean_name} = self.add_weight(name='{clean_name}', shape={shape}, initializer='zeros', trainable={trainable})")
    lines.extend(["", f"    def call({args_str}):", return_stmt])
    return lines


def emit_backend_stateful_class(
    target_framework: str,
    graph: IRGraph,
    class_name: str = "LoweredModule",
) -> str:
    """Emit a stateful object-oriented class definition for `target_framework`.

    Args:
        target_framework (str): Target backend/framework identifier.
        graph (IRGraph): Functional IR graph with parameter inputs and outputs.
        class_name (str): Name of the generated stateful class.

    Returns:
        str: Generated Python source code representing the stateful class.
    """
    state_inputs = [n for n in graph.nodes.values() if (n.op_type == "Input" and n.attributes.get("is_state", False)) or n.op_type == "ReadVariable"]
    non_state_inputs = [n for n in graph.nodes.values() if n.op_type == "Input" and not n.attributes.get("is_state", False)]

    args_str = ", ".join(["self"] + [inp.id for inp in non_state_inputs])
    return_stmt = f"        return {', '.join(graph.outputs)}" if graph.outputs else "        return None"

    fw = target_framework.lower()
    if fw in ("pytorch", "torch"):
        lines = _emit_pytorch_class(class_name, state_inputs, args_str, return_stmt)
    elif fw == "jax":
        lines = _emit_jax_class(class_name, state_inputs, args_str, return_stmt)
    elif fw == "keras":
        lines = _emit_keras_class(class_name, state_inputs, args_str, return_stmt)
    else:
        lines = [
            f"class {class_name}:",
            '    """Generated stateful module from functional IRGraph."""',
            "",
            "    def __init__(self) -> None:",
            "        pass",
            "",
            f"    def __call__({args_str}):",
            return_stmt,
        ]
    nl = chr(10)
    return nl.join(lines) + nl
