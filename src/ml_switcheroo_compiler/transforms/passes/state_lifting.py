"""Canonical State Lifting Pass for functionalizing stateful IR graphs and variable containers."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, IRGraph, IRNode, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeDim,
    ShapeProtocol,
    TypedVariadicCallable,
    is_variadic_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor


def _extract_state_shape(seq: Sequence[ShapeDim]) -> tuple[int, ...]:
    """Convert a shape sequence into a concrete integer shape tuple.

    Args:
        seq (Sequence[ShapeDim]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


StateEntry = OpValue
StateValue = StateEntry


@typing.runtime_checkable
class StatefulContainerProtocol(typing.Protocol):
    """Universal protocol for stateful containers exposing ``parameters()`` and ``buffers()``."""

    def parameters(self) -> Mapping[str, StateEntry]:
        """Return trainable parameter mapping keyed by parameter name.

        Returns:
            Mapping[str, StateEntry]: Trainable parameter dictionary.
        """
        ...

    def buffers(self) -> Mapping[str, StateEntry]:
        """Return non-trainable buffer mapping keyed by buffer name.

        Returns:
            Mapping[str, StateEntry]: Non-trainable buffer dictionary.
        """
        ...


@typing.runtime_checkable
class StateDictContainerProtocol(typing.Protocol):
    """Protocol for containers exposing ``state_dict()`` mapping."""

    def state_dict(self) -> Mapping[str, StateEntry]:
        """Return complete state dictionary mapping names to tensors or nested containers.

        Returns:
            Mapping[str, StateEntry]: State dictionary.
        """
        ...


@typing.runtime_checkable
class MutableStateDictContainerProtocol(typing.Protocol):
    """Protocol for containers supporting in-place state replacement via ``load_state_dict``."""

    def load_state_dict(self, state: Mapping[str, StateEntry]) -> None:
        """Apply updated state entries onto the container.

        Args:
            state (Mapping[str, StateEntry]): Updated state dictionary.
        """
        ...


StateContainerInput = typing.Union[
    StateEntry,
    StatefulContainerProtocol,
    StateDictContainerProtocol,
    "MutableStateDictContainerProtocol",
]


@typing.runtime_checkable
class _StateContainerCallableProtocol(typing.Protocol):
    """Protocol for callable stateful containers accepting ``StateEntry`` arguments."""

    def __call__(self, *args: StateEntry, **kwargs: StateEntry) -> StateEntry:
        """Invoke the container.

        Args:
            *args (StateEntry): Positional arguments.
            **kwargs (StateEntry): Keyword arguments.

        Returns:
            StateEntry: Computed ``StateEntry`` output.
        """
        ...


@typing.runtime_checkable
class _BlockNodesProtocol(typing.Protocol):
    """Protocol for graph/block containers exposing ``nodes``."""

    nodes: list[LogicalNode] | dict[str, LogicalNode]


@typing.runtime_checkable
class _BlockOutputsProtocol(typing.Protocol):
    """Protocol for graph/block containers exposing ``outputs``."""

    outputs: list[str]


@typing.runtime_checkable
class _RequiresGradProtocol(typing.Protocol):
    """Protocol for tensor-like objects exposing ``requires_grad``."""

    requires_grad: bool


@typing.runtime_checkable
class _DTypeProtocol(typing.Protocol):
    """Protocol for tensor-like objects exposing ``dtype``."""

    dtype: DType | str | None


@typing.runtime_checkable
class PureStateCallableProtocol(typing.Protocol):
    """Protocol for pure functional state callables ``(state, *args, **kwargs) -> (out, new_state)``."""

    def __call__(
        self,
        state: Mapping[str, StateEntry],
        *args: StateEntry,
        **kwargs: StateEntry,
    ) -> tuple[StateEntry, dict[str, StateEntry]]:
        """Execute pure state function.

        Args:
            state (Mapping[str, StateEntry]): Input state dictionary.
            *args (StateEntry): Positional arguments.
            **kwargs (StateEntry): Keyword arguments.

        Returns:
            tuple[StateEntry, dict[str, StateEntry]]: Output and updated state dictionary.
        """
        ...


def _iter_container_attrs(container: StateContainerInput) -> list[tuple[str, StateEntry]]:
    """Extract structural attributes from a container via ``StatefulContainerProtocol``.

    ``StateDictContainerProtocol``, or ``Mapping``.

    Args:
        container (StateContainerInput): Container object to inspect.

    Returns:
        list[tuple[str, StateEntry]]: Ordered ``(attribute_name, attribute_value)`` pairs.
    """
    if isinstance(container, StatefulContainerProtocol):
        entries: list[tuple[str, StateEntry]] = list(container.parameters().items())
        entries.extend(container.buffers().items())
        return entries
    if isinstance(container, StateDictContainerProtocol):
        return list(container.state_dict().items())
    if isinstance(container, Mapping):
        return [(str(k), v) for k, v in container.items()]
    return []


def _get_node_items(block: LogicalGraph | _BlockNodesProtocol | None) -> Iterable[tuple[str, LogicalNode]]:
    """Get all ``(node_id, node)`` pairs from a block or graph.

    Args:
        block (LogicalGraph | _BlockNodesProtocol | None): The block or graph to inspect.

    Returns:
        Iterable[tuple[str, LogicalNode]]: Sequence of node ID and ``LogicalNode`` pairs.
    """
    if not isinstance(block, _BlockNodesProtocol):
        return []
    nodes: list[LogicalNode] | Mapping[str, LogicalNode] = block.nodes
    if isinstance(nodes, Mapping):
        return list(nodes.items())
    return [(n.id, n) for n in nodes]


def _lift_node_state(node: LogicalNode, nid: str, block: LogicalGraph | _BlockNodesProtocol) -> bool:
    """Lift stateful ``ReadVariable``/``AssignVariable``/``Assign`` operations into functional I/O.

    Args:
        node (LogicalNode): Node being inspected.
        nid (str): Unique node identifier.
        block (LogicalGraph | _BlockNodesProtocol): Containing block or graph.

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
        if isinstance(block, _BlockOutputsProtocol) and nid not in block.outputs:
            block.outputs.append(nid)
        return True

    return False


def _lift_block(block: LogicalGraph | _BlockNodesProtocol | AttributeValue) -> bool:
    """Recursively lift stateful operations across blocks and nested subgraphs.

    Args:
        block (LogicalGraph | _BlockNodesProtocol | AttributeValue): Block or graph to lift.

    Returns:
        bool: True if any stateful operation was lifted.
    """
    assert isinstance(block, _BlockNodesProtocol)
    block_mod: bool = False
    for nid, node in _get_node_items(block):
        block_mod = _lift_node_state(node, nid, block) or block_mod

        for sub in node.subgraphs.values():
            block_mod = _lift_block(sub) or block_mod
        for raw_attr in node.attributes.values():
            cand_attr: AttributeValue | _BlockNodesProtocol = raw_attr
            if isinstance(cand_attr, _BlockNodesProtocol):
                block_mod = _lift_block(cand_attr) or block_mod
    return block_mod


def _get_val(
    curr_obj: StateContainerInput,
    key: str,
) -> StateEntry:
    """Retrieve value from a dictionary or ``StateDictContainerProtocol`` container.

    Args:
        curr_obj (StateContainerInput): Mapping or protocol container.
        key (str): State entry key.

    Returns:
        StateEntry: Retrieved value or ``None``.
    """
    if isinstance(curr_obj, dict):
        return curr_obj.get(key)
    if not isinstance(curr_obj, Tensor) and isinstance(curr_obj, StateDictContainerProtocol):
        sd_map: Mapping[str, StateEntry] = curr_obj.state_dict()
        return sd_map.get(key)
    if not isinstance(curr_obj, Tensor) and isinstance(curr_obj, StatefulContainerProtocol):
        params: Mapping[str, StateEntry] = curr_obj.parameters()
        if key in params:
            return params[key]
        bufs: Mapping[str, StateEntry] = curr_obj.buffers()
        return bufs.get(key)
    return None


def _set_val(curr_obj: StateContainerInput, key: str, val: StateEntry) -> None:
    """Update value on a dictionary or ``MutableStateDictContainerProtocol`` container.

    Args:
        curr_obj (StateContainerInput): Target dictionary or mutable protocol container.
        key (str): Key or state entry name.
        val (StateEntry): Value to assign.
    """
    if isinstance(curr_obj, dict):
        curr_obj[key] = val
    elif isinstance(curr_obj, MutableStateDictContainerProtocol):
        curr_obj.load_state_dict({key: val})


def _navigate(curr_obj: StateContainerInput, parts: list[str]) -> StateContainerInput:
    """Traverse nested dictionary or object hierarchy along path parts.

    Args:
        curr_obj (StateContainerInput): Root object.
        parts (list[str]): Dot-separated path components.

    Returns:
        StateContainerInput: Navigated nested object.
    """
    curr: StateContainerInput = curr_obj
    for p in parts:
        curr = curr[p] if isinstance(curr, dict) else _get_val(curr, p)
    return curr


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
        """Record a tensor-like entry into either ``params`` or ``buffers`` and update ``identity_map``.

        Args:
            name (str): Fully qualified state entry name.
            val (StateEntry): Tensor or array-like value.
            params (dict[str, StateEntry]): Parameter dictionary to populate.
            buffers (dict[str, StateEntry]): Buffer dictionary to populate.
        """
        cand_val: StateEntry | _RequiresGradProtocol = val
        req_grad = cand_val.requires_grad if isinstance(cand_val, _RequiresGradProtocol) else True
        if req_grad:
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
        if isinstance(attr_val, (Tensor, ShapeProtocol)):
            self._record_tensor_entry(full_name, attr_val, params, buffers)
        elif isinstance(attr_val, dict):
            for k, v in attr_val.items():
                if isinstance(v, (Tensor, ShapeProtocol)):
                    self._record_tensor_entry(f"{full_name}.{k}", v, params, buffers)
        elif (
            not callable(attr_val)
            and not isinstance(attr_val, (int, float, str, bool, list, tuple))
            and attr_val is not None
        ):
            sub_p, sub_b = self.extract_state(attr_val, prefix=f"{full_name}.")
            params.update(sub_p)
            buffers.update(sub_b)

    def extract_state(
        self,
        container: StateContainerInput,
        prefix: str = "",
    ) -> tuple[dict[str, StateEntry], dict[str, StateEntry]]:
        """Extract trainable parameters and non-trainable buffers from a variable container.

        Args:
            container (StateContainerInput): Stateful variable container.
            prefix (str): Prefix path for nested containers.

        Returns:
            tuple[dict[str, StateEntry], dict[str, StateEntry]]: ``(params_dict, buffers_dict)``.
        """
        params: dict[str, StateEntry] = {}
        buffers: dict[str, StateEntry] = {}

        if isinstance(container, StatefulContainerProtocol):
            for p_name, p_val in container.parameters().items():
                full_p = f"{prefix}{p_name}" if prefix else p_name
                params[full_p] = p_val
                self.identity_map[id(p_val)] = full_p
            for b_name, b_val in container.buffers().items():
                full_b = f"{prefix}{b_name}" if prefix else b_name
                buffers[full_b] = b_val
                self.identity_map[id(b_val)] = full_b
            return params, buffers

        entries = _iter_container_attrs(container)
        for attr_name, attr_val in entries:
            if attr_name.startswith("_") and not attr_name.startswith("_buffers"):
                continue
            full_name = f"{prefix}{attr_name}" if prefix else attr_name
            self._extract_attr_entry(full_name, attr_val, params, buffers)

        return params, buffers

    def lift(self, container: StateContainerInput) -> tuple[dict[str, StateEntry], IRGraph]:
        """Lift container parameters and buffers into an ``IRGraph`` with typed ``shape_metadata`` tuples.

        Args:
            container (StateContainerInput): Stateful container instance.

        Returns:
            tuple[dict[str, StateEntry], IRGraph]: Combined state dictionary and functional graph.
        """
        params, buffers = self.extract_state(container)
        combined_state: dict[str, StateEntry] = {}
        combined_state.update(params)
        combined_state.update(buffers)

        graph = IRGraph(name=f"functional_{type(container).__name__}")
        graph.inputs = []
        graph.outputs = []
        for name, tensor in combined_state.items():
            nid = f"param_{name.replace('.', '_')}"
            shape_meta: tuple[int, ...] = (
                _extract_state_shape(tensor.shape) if isinstance(tensor, ShapeProtocol) else ()
            )
            dtype_meta = str(tensor.dtype) if isinstance(tensor, _DTypeProtocol) else "float32"
            is_param = name in params
            node_attrs: dict[str, AttributeValue] = {
                "name": name,
                "param_name": name,
                "is_state": True,
                "is_parameter": is_param,
                "is_buffer": not is_param,
                "dtype": str(dtype_meta),
            }
            node = IRNode(
                id=nid,
                op_type="Input",
                inputs=[],
                shape_metadata=shape_meta,
                attributes=node_attrs,
            )
            graph.nodes[nid] = node
            graph.inputs.append(nid)

        return combined_state, graph

    def functionalize(
        self,
        container: TypedVariadicCallable[OpValue] | StateContainerInput,
    ) -> tuple[dict[str, StateEntry], PureStateCallableProtocol]:
        """Convert a stateful callable container into a pure function accepting and returning state.

        Args:
            container (TypedVariadicCallable[OpValue] | StateContainerInput): Stateful callable container.

        Returns:
            tuple[dict[str, StateEntry], PureStateCallableProtocol]: Initial state dictionary and pure function.
        """
        params, buffers = self.extract_state(container)
        current_state: dict[str, StateEntry] = {}
        current_state.update(params)
        current_state.update(buffers)
        state_container: StateContainerInput = container

        def pure_fn(
            state: Mapping[str, StateEntry],
            *args: StateEntry,
            **kwargs: StateEntry,
        ) -> tuple[StateEntry, dict[str, StateEntry]]:
            """Execute container purely with explicit input state and restored attributes.

            Args:
                state (Mapping[str, StateEntry]): Input state dictionary.
                *args (StateEntry): Positional arguments.
                **kwargs (StateEntry): Keyword arguments.

            Returns:
                tuple[StateEntry, dict[str, StateEntry]]: Output and updated state dictionary.

            Raises:
                TypeError: If ``container`` is not callable.
            """
            backup: dict[str, StateEntry] = {}
            for k, v in state.items():
                parts = k.split(".")
                parent = _navigate(state_container, parts[:-1])
                backup[k] = _get_val(parent, parts[-1])
                _set_val(parent, parts[-1], v)

            try:
                if not callable(container):
                    raise TypeError(f"Container {type(container)} is not callable.")
                _callable_fn_1: OpValue = container
                assert is_variadic_callable(_callable_fn_1)
                out: StateEntry = _callable_fn_1(*args, **kwargs)

                new_state: dict[str, StateEntry] = {}
                for k in state:
                    parts = k.split(".")
                    parent = _navigate(state_container, parts[:-1])
                    new_state[k] = _get_val(parent, parts[-1])

                return out, new_state
            finally:
                for k, v in backup.items():
                    parts = k.split(".")
                    parent = _navigate(state_container, parts[:-1])
                    _set_val(parent, parts[-1], v)

        return current_state, pure_fn

    def run(self, graph: IRGraph) -> bool:
        """Run in-place state lifting pass on ``IRGraph``.

        Args:
            graph (IRGraph): Target IR graph.

        Returns:
            bool: True if graph was modified, False otherwise.
        """
        return _lift_block(graph)


def flatten_state_dict(
    state_dict: Mapping[str, StateValue],
    prefix: str = "",
) -> dict[str, StateValue]:
    """Flatten a nested state dictionary into a dot-delimited flat map.

    Args:
        state_dict (Mapping[str, StateValue]): Nested state dictionary.
        prefix (str): Key prefix for recursion.

    Returns:
        dict[str, StateValue]: Flattened state map.
    """
    flat: dict[str, StateValue] = {}
    for k, v in state_dict.items():
        new_key: str = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            sub_map: dict[str, StateValue] = {str(sk): sv for sk, sv in v.items()}
            flat.update(flatten_state_dict(sub_map, new_key))
        else:
            flat[new_key] = v
    return flat


def unflatten_state_dict(
    flat_state: dict[str, StateValue],
) -> dict[str, StateValue | dict[str, StateValue]]:
    """Unflatten a dot-delimited state dictionary back into a nested structure.

    Args:
        flat_state (dict[str, StateValue]): Flattened state map.

    Returns:
        dict[str, StateValue | dict[str, StateValue]]: Nested state dictionary.
    """
    nested: dict[str, StateValue | dict[str, StateValue]] = {}
    for k, v in flat_state.items():
        parts: list[str] = k.split(".")
        d: dict[str, StateValue | dict[str, StateValue]] = nested
        for part in parts[:-1]:
            existing = d.get(part)
            if isinstance(existing, dict):
                d = existing
            else:
                sub_dict: dict[str, StateValue | dict[str, StateValue]] = {}
                d[part] = sub_dict
                d = sub_dict
        d[parts[-1]] = v
    return nested


def _get_nodes(block: LogicalGraph) -> Iterable[LogicalNode]:
    """Extract all ``LogicalNode`` instances from a block or graph.

    Args:
        block (LogicalGraph): The block or graph to inspect.

    Returns:
        Iterable[LogicalNode]: Sequence of ``LogicalNode`` instances.
    """
    return [node for _, node in _get_node_items(block)]


def _lift_node(node: LogicalNode, block: LogicalGraph) -> bool:
    """Lift a single ``ReadVariable``/``AssignVariable``/``Assign`` node in place.

    Args:
        node (LogicalNode): Target node.
        block (LogicalGraph): Containing block or graph.

    Returns:
        bool: True if the node was modified.
    """
    return _lift_node_state(node, node.id, block)


def _lift_block_ir(block: LogicalGraph) -> bool:
    """Recursively lift stateful operations across a block and its subgraphs.

    Args:
        block (LogicalGraph): Target block or graph.

    Returns:
        bool: True if any node was lifted.
    """
    return _lift_block(block)


def state_lifting_pass(graph: IRGraph) -> bool:
    """In-place pass to lift state to pure functional inputs.

    Args:
        graph (IRGraph): Target graph.

    Returns:
        bool: True if the graph was modified.
    """
    return StateLiftingPass().run(graph)


def lift_state_pass(graph: IRGraph) -> bool:
    """Alias for ``state_lifting_pass`` lifting implicit state into functional I/O.

    Args:
        graph (IRGraph): Target IRGraph.

    Returns:
        bool: True if the graph was modified.
    """
    return state_lifting_pass(graph)


def lift_module_state(container: StateContainerInput) -> tuple[dict[str, StateValue], IRGraph]:
    """Lift a variable container into pure function parameters and an ``IRGraph``.

    Args:
        container (StateContainerInput): Stateful variable container.

    Returns:
        tuple[dict[str, StateValue], IRGraph]: Extracted state map and functional ``IRGraph``.
    """
    return StateLiftingPass().lift(container)


def lift_state(
    container: TypedVariadicCallable[OpValue] | StateContainerInput,
) -> tuple[dict[str, StateValue], PureStateCallableProtocol]:
    """Functionalize a callable container into pure signature ``(state, *args) -> (out, new_state)``.

    Args:
        container (TypedVariadicCallable[OpValue] | StateContainerInput): Callable stateful container.

    Returns:
        tuple[dict[str, StateValue], PureStateCallableProtocol]: Initial state dict and pure callable.
    """
    return StateLiftingPass().functionalize(container)


def functionalize_assign_graph(graph: LogicalGraph, state_vars: list[str]) -> LogicalGraph:
    """Lift mutable ``Assign`` nodes on ``state_vars`` into pure functional graph outputs.

    Args:
        graph (LogicalGraph): Input graph containing state ``Assign`` nodes.
        state_vars (list[str]): Node IDs representing initial state variables.

    Returns:
        LogicalGraph: Functionalized graph with updated state outputs appended.
    """
    from ml_switcheroo_compiler.core.state_manager import lift_state as _lift_assign_state

    return _lift_assign_state(graph, state_vars)


__all__ = [
    "MutableStateDictContainerProtocol",
    "PureStateCallableProtocol",
    "StateContainerInput",
    "StateDictContainerProtocol",
    "StateEntry",
    "StateLiftingPass",
    "StateValue",
    "StatefulContainerProtocol",
    "_get_node_items",
    "_get_nodes",
    "_get_val",
    "_lift_block",
    "_lift_block_ir",
    "_lift_node",
    "_lift_node_state",
    "_navigate",
    "_set_val",
    "flatten_state_dict",
    "functionalize_assign_graph",
    "lift_module_state",
    "lift_state",
    "lift_state_pass",
    "state_lifting_pass",
    "unflatten_state_dict",
]
