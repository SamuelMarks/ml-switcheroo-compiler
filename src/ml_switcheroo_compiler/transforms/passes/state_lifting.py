"""State Lifting Pass for functionalization of stateful IR graphs and variable containers."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from typing import cast

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.shape import TensorOrScalar
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
    """Lift stateful ReadVariable/AssignVariable/Assign operations into pure functional inputs/outputs.

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

    if node.op_type in ("AssignVariable", "Assign"):
        var_name = str(node.attributes.get("variable_name", f"var_{nid}"))
        node.op_type = "Output"
        if len(node.inputs) > 1:
            node.inputs = [node.inputs[1]]
        node.attributes["name"] = f"{var_name}_out"
        node.attributes["stop_gradient"] = False
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


def _get_val(curr_obj: TensorOrScalar, key: str) -> StateEntry:
    """Retrieve value from dictionary or object attribute.

    Args:
        curr_obj (TensorOrScalar): Object or dictionary container.
        key (str): Attribute name or dictionary key.

    Returns:
        StateEntry: Retrieved value or None.
    """
    if isinstance(curr_obj, dict):
        return curr_obj.get(key)
    return getattr(curr_obj, key, None)


def _set_val(curr_obj: TensorOrScalar, key: str, val: StateEntry) -> None:
    """Set value on dictionary or object attribute.

    Args:
        curr_obj (TensorOrScalar): Target container.
        key (str): Key or attribute name.
        val (StateEntry): Value to assign.
    """
    if isinstance(curr_obj, dict):
        curr_obj[key] = val
    else:
        setattr(curr_obj, key, val)


def _navigate(curr_obj: TensorOrScalar, parts: list[str]) -> TensorOrScalar:
    """Traverse nested dictionary or object hierarchy along path parts.

    Args:
        curr_obj (TensorOrScalar): Root object.
        parts (list[str]): Dot-separated path components.

    Returns:
        TensorOrScalar: Navigated nested object.
    """
    for p in parts:
        if isinstance(curr_obj, dict):
            curr_obj = curr_obj[p]
        else:
            curr_obj = getattr(curr_obj, p)
    return curr_obj


class StateLiftingPass:
    """Graph and container state lifting pass extracting trainable and non-trainable tensor fields."""

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
        """Extract state from a single attribute value.

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

    def extract_state(self, container: TensorOrScalar, prefix: str = "") -> tuple[dict[str, StateEntry], dict[str, StateEntry]]:
        """Extract trainable parameters and non-trainable buffers from a generic variable container.

        Args:
            container (TensorOrScalar): Stateful variable container.
            prefix (str): Prefix path for nested containers.

        Returns:
            tuple[dict[str, StateEntry], dict[str, StateEntry]]: (params_dict, buffers_dict).
        """
        params: dict[str, StateEntry] = {}
        buffers: dict[str, StateEntry] = {}

        if not hasattr(container, "__dict__"):
            return params, buffers

        for attr_name, attr_val in container.__dict__.items():
            if attr_name.startswith("_") and not attr_name.startswith("_buffers"):
                continue
            full_name = f"{prefix}{attr_name}" if prefix else attr_name
            self._extract_attr_entry(full_name, attr_val, params, buffers)

        return params, buffers

    def lift(self, container: TensorOrScalar) -> tuple[dict[str, StateEntry], IRGraph]:
        """Lift container parameters and buffers into an IRGraph with functional input nodes.

        Args:
            container (TensorOrScalar): Stateful container instance.

        Returns:
            tuple[dict[str, StateEntry], IRGraph]: Combined state dictionary and functional graph.
        """
        params, buffers = self.extract_state(container)
        combined_state = {**params, **buffers}

        graph = IRGraph(name=f"functional_{container.__class__.__name__}")
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
                    "param_name": name,
                    "is_state": True,
                    "is_parameter": is_param,
                    "is_buffer": not is_param,
                    "dtype": str(dtype_meta),
                },
            )
            graph.nodes[nid] = node
            graph.inputs.append(nid)

        return combined_state, graph

    def functionalize(self, container: Callable[..., TensorOrScalar]) -> tuple[dict[str, StateEntry], Callable[..., tuple[StateEntry, dict[str, StateEntry]]]]:
        """Convert a stateful callable container into a pure function accepting and returning state.

        Args:
            container (Callable[..., TensorOrScalar]): The stateful callable container.

        Returns:
            tuple[dict[str, StateEntry], Callable[..., tuple[StateEntry, dict[str, StateEntry]]]]: Initial state dictionary and pure function.
        """
        params, buffers = self.extract_state(container)
        current_state = {**params, **buffers}

        def pure_fn(state: dict[str, StateEntry], *args: StateEntry, **kwargs: StateEntry) -> tuple[StateEntry, dict[str, StateEntry]]:
            """Pure execution function with explicit state passing.

            Args:
                state (dict[str, StateEntry]): Input state dictionary.
                *args (StateEntry): Positional arguments.
                **kwargs (StateEntry): Keyword arguments.

            Returns:
                tuple[StateEntry, dict[str, StateEntry]]: Output and updated state dictionary.
            """
            backup: dict[str, StateEntry] = {}
            for k, v in state.items():
                parts = k.split(".")
                parent = _navigate(container, parts[:-1])
                backup[k] = _get_val(parent, parts[-1])
                _set_val(parent, parts[-1], v)

            try:
                if callable(container):
                    out = container(*args, **kwargs)
                else:
                    raise TypeError(f"Container {type(container)} is not callable.")

                new_state: dict[str, StateEntry] = {}
                for k in state:
                    parts = k.split(".")
                    parent = _navigate(container, parts[:-1])
                    new_state[k] = _get_val(parent, parts[-1])

                return cast(tuple[StateEntry, dict[str, StateEntry]], (out, new_state))
            finally:
                for k, v in backup.items():
                    parts = k.split(".")
                    parent = _navigate(container, parts[:-1])
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
        bool: True if the graph was modified.
    """
    return StateLiftingPass().run(graph)
