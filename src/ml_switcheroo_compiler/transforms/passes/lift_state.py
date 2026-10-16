# ruff: noqa: E402
"""Lift State pass."""

import typing
from collections.abc import Iterable
from typing import Callable, cast

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import EagerValue
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

StateValue = typing.Union[int, float, str, bool, list["StateValue"], tuple["StateValue", ...], dict[str, "StateValue"], None]


def flatten_state_dict(state_dict: dict[str, typing.Union[StateValue, dict[str, StateValue]]], prefix: str = "") -> dict[str, StateValue]:
    """Flatten a nested state dictionary (like flax.nnx.State) into a flat map.

    Args:
        state_dict: Nested state dict
        prefix: Prefix for keys

    Returns:
        Flattened state map
    """
    flat: dict[str, StateValue] = {}
    for k, v in state_dict.items():
        new_key: str = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_state_dict(v, new_key))
        else:
            flat[new_key] = v
    return flat


def unflatten_state_dict(flat_state: dict[str, StateValue]) -> dict[str, typing.Union[StateValue, dict[str, StateValue]]]:
    """Unflatten a state dict back to nested structure.

    Args:
        flat_state: Flattened state map

    Returns:
        Nested state dict
    """
    nested: dict[str, typing.Union[StateValue, dict[str, StateValue]]] = {}
    for k, v in flat_state.items():
        parts: list[str] = k.split(".")
        d: dict[str, StateValue] = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = typing.cast(dict[str, StateValue], d[part])
        d[parts[-1]] = v
    return nested


def _get_nodes(block: LogicalGraph) -> Iterable[LogicalNode]:
    """Evaluate _get_nodes operation.

    Args:
        block (LogicalGraph): The block parameter.

    Returns:
            Iterable[LogicalNode]: Result.
    """
    nodes: typing.Union[list[LogicalNode], dict[str, LogicalNode]] = getattr(block, "nodes", {})
    if isinstance(nodes, dict):
        return nodes.values()
    return nodes


def _lift_node(node: LogicalNode, block: LogicalGraph) -> bool:
    """Evaluate _lift_node operation.

    Args:
        node (LogicalNode): The node parameter.
        block (LogicalGraph): The block parameter.

    Returns:
        bool: Result.
    """
    if node.op_type == "ReadVariable":
        node.op_type = "Input"
        return True
    if node.op_type in ("AssignVariable", "Assign"):
        node.op_type = "Output"
        if len(node.inputs) > 1:
            # For Assign(var, value), we only want to output the new value
            node.inputs = [node.inputs[1]]

        # VERY IMPORTANT: When integrating state updates into the autodiff gradient tape,
        # we must ensure that the output node inherits a gradient passthrough hook so
        # that reverse-mode AD accurately tracks mutating state backward.
        node.attributes["stop_gradient"] = False
        node.attributes["is_state_update"] = True

        if hasattr(block, "outputs") and node.id not in block.outputs:
            block.outputs.append(node.id)
        return True
    return False


def _lift_block_ir(block: LogicalGraph) -> bool:
    """Evaluate _lift_block_ir operation.

    Args:
        block (LogicalGraph): The block parameter.

    Returns:
        bool: Result.
    """
    mod: bool = False
    for node in _get_nodes(block):
        mod = _lift_node(node, block) or mod
        for sub in getattr(node, "subgraphs", {}).values():
            mod = _lift_block_ir(sub) or mod
        for attr_val in node.attributes.values():
            if hasattr(attr_val, "nodes"):
                mod = _lift_block_ir(typing.cast(LogicalGraph, attr_val)) or mod
    return mod


def lift_state_pass(graph: IRGraph) -> bool:
    """In-place pass to lift implicit state into functional I/O.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    return _lift_block_ir(graph)


def lift_module_state(module: EagerValue) -> tuple[dict[str, StateValue], IRGraph]:
    """Functionalize a class-based stateful module (e.g. nn.Module) into pure function parameters and an IRGraph.

    Args:
        module: Stateful object or PyTorch nn.Module instance.

    Returns:
        tuple[dict[str, StateValue], IRGraph]: Extracted parameters map and pure functional IRGraph.
    """
    params: dict[str, StateValue] = {}

    if hasattr(module, "named_parameters"):
        named_params = getattr(module, "named_parameters")()
        for k, v in named_params:
            params[k] = v
    elif hasattr(module, "state_dict"):
        sd = getattr(module, "state_dict")()
        for k, v in sd.items():
            params[k] = v
    elif hasattr(module, "__dict__"):
        for k, v in module.__dict__.items():
            if not k.startswith("_") and not callable(v):
                params[k] = v

    graph = IRGraph()
    for param_name, param_val in params.items():
        clean_id = f"param_{param_name.replace('.', '_')}"
        shape = tuple(getattr(param_val, "shape", ())) if hasattr(param_val, "shape") else ()
        graph.nodes[clean_id] = IRNode(
            id=clean_id,
            op_type="Input",
            attributes={"is_state": True, "param_name": param_name},
            shape_metadata={"shape": shape},
        )

    return params, graph


def lift_state(
    module: Callable[..., EagerValue],
) -> tuple[dict[str, StateValue], typing.Callable[..., tuple[StateValue, dict[str, StateValue]]]]:
    """Functionalize a class-based stateful module into pure function signature: (params, inputs) -> (outputs, updated_params).

    Args:
        module: Class-based stateful module (e.g. PyTorch nn.Module).

    Returns:
        tuple[dict[str, StateValue], typing.Callable[..., tuple[StateValue, dict[str, StateValue]]]]: Extracted parameter dict and pure functional callable.
    """
    params, _ = lift_module_state(module)

    def pure_fn(
        current_params: dict[str, StateValue],
        *args: StateValue,
        **kwargs: StateValue,
    ) -> tuple[StateValue, dict[str, StateValue]]:
        """Pure functional execution of the lifted module.

        Args:
            current_params (dict[str, StateValue]): Parameters passed functionally.
            *args (StateValue): Positional module arguments.
            **kwargs (StateValue): Keyword module arguments.

        Returns:
            tuple[StateValue, dict[str, StateValue]]: Module forward outputs and updated state dict.
        """
        backup: dict[str, StateValue] = {}
        for k, v in current_params.items():
            if hasattr(module, k):
                backup[k] = getattr(module, k)
            setattr(module, k, v)

        try:
            if callable(module):
                output = module(*args, **kwargs)
            elif hasattr(module, "forward"):
                output = getattr(module, "forward")(*args, **kwargs)
            else:
                output = None
            updated_params: dict[str, StateValue] = {k: getattr(module, k, v) for k, v in current_params.items()}
        finally:
            for k, v in backup.items():
                setattr(module, k, v)

        return cast(tuple[StateValue, dict[str, StateValue]], (output, updated_params))

    return params, pure_fn


from ml_switcheroo_compiler.transforms.passes.state_lifting import StateLiftingPass  # noqa: F401
