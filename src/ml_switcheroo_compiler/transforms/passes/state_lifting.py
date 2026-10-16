"""State Lifting Pass for functionalization of stateful OOP model representations."""

import typing
from collections.abc import Callable, Iterable
from typing import cast

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import EagerValue
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
        var_name = str(node.attributes.get("variable_name", f"var_{nid}"))
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


StateEntry = typing.Union[Tensor, int, float, str, bool, list["StateEntry"], tuple["StateEntry", ...], dict[str, "StateEntry"], None]


def _get_val(curr_obj: EagerValue, key: str) -> StateEntry:
    """Retrieve value from dictionary or object attribute.

    Args:
        curr_obj: Object or dictionary container.
        key (str): Attribute name or dictionary key.

    Returns:
        StateEntry: Retrieved value or None.
    """
    if isinstance(curr_obj, dict):
        return curr_obj.get(key)
    return getattr(curr_obj, key, None)


def _set_val(curr_obj: EagerValue, key: str, val: StateEntry) -> None:
    """Set value on dictionary or object attribute.

    Args:
        curr_obj: Target container.
        key (str): Key or attribute name.
        val (StateEntry): Value to assign.
    """
    if isinstance(curr_obj, dict):
        curr_obj[key] = val
    else:
        setattr(curr_obj, key, val)


def _navigate(curr_obj: EagerValue, parts: list[str]) -> EagerValue:
    """Traverse nested dictionary or object hierarchy along path parts.

    Args:
        curr_obj: Root object.
        parts (list[str]): Dot-separated path components.

    Returns:
        dict: Navigated nested object.
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

    def _record_tensor_entry(
        self,
        name: str,
        val: StateEntry,
        params: dict[str, StateEntry],
        buffers: dict[str, StateEntry],
    ) -> None:
        """Record a tensor-like entry into either params or buffers and update identity_map.

        Args:
            name (str): Fully qualified state entry name.
            val (StateEntry): Tensor or array-like value.
            params (dict[str, StateEntry]): Parameter dictionary to populate.
            buffers (dict[str, StateEntry]): Buffer dictionary to populate.
        """
        if getattr(val, "requires_grad", True):
            params[name] = val
        else:
            buffers[name] = val
        self.identity_map[id(val)] = name

    def _extract_attr_entry(
        self,
        full_name: str,
        attr_val: StateEntry,
        params: dict[str, StateEntry],
        buffers: dict[str, StateEntry],
    ) -> None:
        """Extract state from a single reflected attribute value.

        Args:
            full_name (str): Qualified attribute name.
            attr_val (StateEntry): Attribute value to inspect.
            params (dict[str, StateEntry]): Parameter dictionary to populate.
            buffers (dict[str, StateEntry]): Buffer dictionary to populate.
        """
        if isinstance(attr_val, Tensor) or hasattr(attr_val, "shape"):
            self._record_tensor_entry(full_name, attr_val, params, buffers)
        elif isinstance(attr_val, dict):
            for k, v in attr_val.items():
                if isinstance(v, Tensor) or hasattr(v, "shape"):
                    self._record_tensor_entry(f"{full_name}.{k}", v, params, buffers)
        elif hasattr(attr_val, "__dict__") and not callable(attr_val):
            sub_p, sub_b = self.extract_state(attr_val, prefix=f"{full_name}.")
            params.update(sub_p)
            buffers.update(sub_b)

    def extract_state(self, module: EagerValue, prefix: str = "") -> tuple[dict[str, StateEntry], dict[str, StateEntry]]:
        """Reliably extract parameters and buffers from stateful OOP model representations.

        Handles nested submodules, custom parameter dictionaries, and buffer assignments
        without breaking tensor identity.

        Args:
            module: Stateful OOP model (e.g. PyTorch nn.Module or custom class).
            prefix (str): Prefix path for nested submodules.

        Returns:
            tuple[dict[str, StateEntry], dict[str, StateEntry]]: (params_dict, buffers_dict).
        """
        params: dict[str, StateEntry] = {}
        buffers: dict[str, StateEntry] = {}

        if hasattr(module, "named_parameters") and callable(getattr(module, "named_parameters")):
            for name, param in getattr(module, "named_parameters")():
                full_name = f"{prefix}{name}" if prefix else name
                params[full_name] = param
                self.identity_map[id(param)] = full_name

        if hasattr(module, "named_buffers") and callable(getattr(module, "named_buffers")):
            for name, buf in getattr(module, "named_buffers")():
                full_name = f"{prefix}{name}" if prefix else name
                buffers[full_name] = buf
                self.identity_map[id(buf)] = full_name

        if params or buffers or not hasattr(module, "__dict__"):
            return params, buffers

        for attr_name, attr_val in module.__dict__.items():
            if attr_name.startswith("_") and not attr_name.startswith("_buffers"):
                continue
            full_name = f"{prefix}{attr_name}" if prefix else attr_name
            self._extract_attr_entry(full_name, attr_val, params, buffers)

        return params, buffers

    def lift(self, module: EagerValue) -> tuple[dict[str, StateEntry], IRGraph]:
        """Lift module parameters and buffers into an IRGraph with functional input nodes.

        Args:
            module: Stateful model instance.

        Returns:
            tuple[dict[str, StateEntry], IRGraph]: Combined state dictionary and functional graph.
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

    def functionalize(self, module: Callable[..., EagerValue]) -> tuple[dict[str, StateEntry], Callable[..., tuple[StateEntry, dict[str, StateEntry]]]]:
        """Convert a stateful module into a pure function accepting and returning state.

        Args:
            module: The stateful model module.

        Returns:
            tuple[dict[str, StateEntry], Callable]: Initial state dictionary and pure function.
        """
        params, buffers = self.extract_state(module)
        current_state = {**params, **buffers}

        def pure_fn(state: dict[str, StateEntry], *args: StateEntry, **kwargs: StateEntry) -> tuple[StateEntry, dict[str, StateEntry]]:
            """Pure forward execution function with explicit state passing.

            Args:
                state: The state parameter.
                *args: The args parameter.
                **kwargs: The kwargs parameter.
            """
            backup: dict[str, StateEntry] = {}
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
                elif hasattr(module, "forward") and callable(getattr(module, "forward")):
                    out = getattr(module, "forward")(*args, **kwargs)
                else:
                    raise TypeError(f"Module {type(module)} is not callable and has no forward method.")

                # Capture updated state (buffers might have been mutated)
                new_state: dict[str, StateEntry] = {}
                for k in state:
                    parts = k.split(".")
                    parent = _navigate(module, parts[:-1])
                    new_state[k] = _get_val(parent, parts[-1])

                return cast(tuple[StateEntry, dict[str, StateEntry]], (out, new_state))
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
