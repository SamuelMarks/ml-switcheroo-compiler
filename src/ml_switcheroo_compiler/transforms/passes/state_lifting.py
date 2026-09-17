# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""State Lifting Pass for functionalization of stateful OOP model representations."""

import typing
from collections.abc import Callable, Iterable

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _get_node_items(block: LogicalGraph) -> Iterable[tuple[str, LogicalNode]]:
    """Get all (node_id, node) pairs from a block or graph.

    Args:
        block (LogicalGraph): The block or graph to inspect.

    Returns:
        Iterable[tuple[str, LogicalNode]]: Sequence of node ID and LogicalNode pairs.
    """
    if not hasattr(block, "nodes"):
        return []
    nodes: typing.Union[list[LogicalNode], dict[str, LogicalNode]] = block.nodes
    if isinstance(nodes, dict):
        return list(nodes.items())
    return [(n.id, n) for n in nodes]


def _lift_node_state(node: LogicalNode, nid: str, block: LogicalGraph) -> bool:
    """Lift stateful ReadVariable/AssignVariable operations into pure functional inputs/outputs.

    Args:
        node (LogicalNode): Node being inspected.
        nid (str): Unique node identifier.
        block (LogicalGraph): Containing block or graph.

    Returns:
        bool: True if node was modified, False otherwise.
    """
    if node.op_type == "ReadVariable":
        var_name: str = str(node.attributes.get("variable_name", f"var_{nid}"))
        node.op_type = "Input"
        node.attributes["name"] = var_name
        node.attributes["is_state"] = True
        return True

    if node.op_type == "AssignVariable":
        var_name: str = str(node.attributes.get("variable_name", f"var_{nid}"))
        node.op_type = "Output"
        node.attributes["name"] = f"{var_name}_out"
        node.attributes["is_state_update"] = True
        if hasattr(block, "outputs") and nid not in block.outputs:
            block.outputs.append(nid)
        return True

    return False


def _lift_block(block: LogicalGraph) -> bool:
    """Recursively lift stateful operations across blocks and nested subgraphs.

    Args:
        block (LogicalGraph): Block or graph to lift.

    Returns:
        bool: True if any stateful operation was lifted.
    """
    block_mod: bool = False
    for nid, node in _get_node_items(block):
        block_mod = _lift_node_state(node, nid, block) or block_mod

        for sub in getattr(node, "subgraphs", {}).values():
            block_mod = _lift_block(sub) or block_mod
        for attr_val in node.attributes.values():
            if hasattr(attr_val, "nodes"):
                block_mod = _lift_block(typing.cast(LogicalGraph, attr_val)) or block_mod
    return block_mod


def _get_val(curr_obj: object, key: str) -> object:
    """Retrieve value from dictionary or object attribute.

    Args:
        curr_obj (object): Object or dictionary container.
        key (str): Attribute name or dictionary key.

    Returns:
        object: Retrieved value or None.
    """
    if isinstance(curr_obj, dict):
        return curr_obj.get(key)
    return getattr(curr_obj, key, None)


def _set_val(curr_obj: object, key: str, val: object) -> None:
    """Set value on dictionary or object attribute.

    Args:
        curr_obj (object): Target container.
        key (str): Key or attribute name.
        val (object): Value to assign.
    """
    if isinstance(curr_obj, dict):
        curr_obj[key] = val
    else:
        setattr(curr_obj, key, val)


def _navigate(curr_obj: object, parts: list[str]) -> object:
    """Traverse nested dictionary or object hierarchy along path parts.

    Args:
        curr_obj (object): Root object.
        parts (list[str]): Dot-separated path components.

    Returns:
        object: Navigated nested object.
    """
    for p in parts:
        if isinstance(curr_obj, dict):
            curr_obj = curr_obj[p]
        else:
            curr_obj = getattr(curr_obj, p)
    return curr_obj


class StateLiftingPass:
    """Harden state lifting pass extracting parameters and buffers without breaking tensor identity."""

    def __init__(self) -> None:
        """Initialize StateLiftingPass."""
        self.identity_map: dict[int, str] = {}

    def extract_state(self, module: object, prefix: str = "") -> tuple[dict[str, object], dict[str, object]]:
        """Reliably extract parameters and buffers from stateful OOP model representations.

        Handles nested submodules, custom parameter dictionaries, and buffer assignments
        without breaking tensor identity.

        Args:
            module (object): Stateful OOP model (e.g. PyTorch nn.Module or custom class).
            prefix (str): Prefix path for nested submodules.

        Returns:
            tuple[dict[str, object], dict[str, object]]: (params_dict, buffers_dict).
        """
        params: dict[str, object] = {}
        buffers: dict[str, object] = {}

        # 1. Direct PyTorch-style named_parameters and named_buffers if available
        if hasattr(module, "named_parameters") and callable(module.named_parameters):
            for name, param in module.named_parameters():
                full_name = f"{prefix}{name}" if prefix else name
                params[full_name] = param
                self.identity_map[id(param)] = full_name

        if hasattr(module, "named_buffers") and callable(module.named_buffers):
            for name, buf in module.named_buffers():
                full_name = f"{prefix}{name}" if prefix else name
                buffers[full_name] = buf
                self.identity_map[id(buf)] = full_name

        if params or buffers:
            return params, buffers

        # 2. General reflection over object attributes
        if hasattr(module, "__dict__"):
            for attr_name, attr_val in module.__dict__.items():
                if attr_name.startswith("_") and not attr_name.startswith("_buffers"):
                    continue

                full_name = f"{prefix}{attr_name}" if prefix else attr_name

                # Check if attribute is tensor parameter or buffer
                if isinstance(attr_val, Tensor) or hasattr(attr_val, "shape"):
                    requires_grad = getattr(attr_val, "requires_grad", True)
                    if requires_grad:
                        params[full_name] = attr_val
                    else:
                        buffers[full_name] = attr_val
                    self.identity_map[id(attr_val)] = full_name

                # Check for nested parameter dictionaries
                elif isinstance(attr_val, dict):
                    for k, v in attr_val.items():
                        nested_name = f"{full_name}.{k}"
                        if isinstance(v, Tensor) or hasattr(v, "shape"):
                            if getattr(v, "requires_grad", True):
                                params[nested_name] = v
                            else:
                                buffers[nested_name] = v
                            self.identity_map[id(v)] = nested_name

                # Check for nested submodules
                elif hasattr(attr_val, "__dict__") and not callable(attr_val):
                    sub_p, sub_b = self.extract_state(attr_val, prefix=f"{full_name}.")
                    params.update(sub_p)
                    buffers.update(sub_b)

        return params, buffers

    def lift(self, module: object) -> tuple[dict[str, object], IRGraph]:
        """Lift module parameters and buffers into an IRGraph with functional input nodes.

        Args:
            module (object): Stateful model instance.

        Returns:
            tuple[dict[str, object], IRGraph]: Combined state dictionary and functional graph.
        """
        params, buffers = self.extract_state(module)
        combined_state = {**params, **buffers}

        graph = IRGraph(name=f"functional_{module.__class__.__name__}")
        graph.inputs = []
        graph.outputs = []
        for name, tensor in combined_state.items():
            nid = f"param_{name.replace('.', '_')}"
            shape_meta = getattr(tensor, "shape", ())
            dtype_meta = getattr(tensor, "dtype", "float32")
            is_param = name in params
            node = IRNode(
                id=nid,
                op_type="Input",
                inputs=[],
                shape_metadata={"shape": tuple(shape_meta)},
                attributes={
                    "name": name,
                    "is_state": True,
                    "is_parameter": is_param,
                    "is_buffer": not is_param,
                    "dtype": str(dtype_meta),
                },
            )
            graph.nodes[nid] = node
            graph.inputs.append(nid)

        return combined_state, graph

    def functionalize(self, module: object) -> tuple[dict[str, object], Callable[..., tuple[object, dict[str, object]]]]:
        """Convert a stateful module into a pure function accepting and returning state.

        Args:
            module (object): The stateful model module.

        Returns:
            tuple[dict[str, object], Callable]: Initial state dictionary and pure function.
        """
        params, buffers = self.extract_state(module)
        current_state = {**params, **buffers}

        def pure_fn(state: dict[str, object], *args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            """Pure forward execution function with explicit state passing."""
            backup: dict[str, object] = {}
            # Apply state to module
            for k, v in state.items():
                parts = k.split(".")
                parent = _navigate(module, parts[:-1])
                backup[k] = _get_val(parent, parts[-1])
                _set_val(parent, parts[-1], v)

            try:
                # Call module forward/__call__
                if callable(module):
                    out = module(*args, **kwargs)
                elif hasattr(module, "forward") and callable(module.forward):
                    out = module.forward(*args, **kwargs)
                else:
                    raise TypeError(f"Module {type(module)} is not callable and has no forward method.")

                # Capture updated state (buffers might have been mutated)
                new_state: dict[str, object] = {}
                for k in state:
                    parts = k.split(".")
                    parent = _navigate(module, parts[:-1])
                    new_state[k] = _get_val(parent, parts[-1])

                return out, new_state
            finally:
                # Restore original attributes
                for k, v in backup.items():
                    parts = k.split(".")
                    parent = _navigate(module, parts[:-1])
                    _set_val(parent, parts[-1], v)

        return current_state, pure_fn

    def run(self, graph: IRGraph) -> bool:
        """Run in-place state lifting pass on IRGraph.

        Args:
            graph (IRGraph): The IR graph.

        Returns:
            bool: True if graph was modified, False otherwise.
        """
        return _lift_block(graph)


def state_lifting_pass(graph: IRGraph) -> bool:
    """In-place pass to lift state to pure functional inputs.

    Args:
        graph (IRGraph): The graph parameter for the operation.

    Returns:
        bool: A boolean indicating the result of the check.
    """
    return StateLiftingPass().run(graph)
