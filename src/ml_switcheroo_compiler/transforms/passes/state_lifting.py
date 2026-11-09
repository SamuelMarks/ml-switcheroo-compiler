"""State Lifting Pass for functionalization of stateful IR graphs and variable containers."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TensorOrScalar,
    TypedVariadicCallable,
)
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _extract_state_shape(seq: Sequence[int | str | None]) -> tuple[int, ...]:
    """Convert a shape sequence into a concrete integer shape tuple.

    Args:
        seq (Sequence[int | str | None]): Shape sequence.

    Returns:
        tuple[int, ...]: Concrete integer shape tuple.
    """
    return tuple(int(d) for d in seq if isinstance(d, int))


StateEntry = TensorOrScalar


@typing.runtime_checkable
class _StateContainerCallableProtocol(typing.Protocol):
    """Protocol for callable stateful containers accepting `StateEntry` arguments."""

    def __call__(self, *args: StateEntry, **kwargs: StateEntry) -> StateEntry:
        """Invoke the container.

        Args:
            *args (StateEntry): Positional arguments.
            **kwargs (StateEntry): Keyword arguments.

        Returns:
            StateEntry: Result.
        """
        ...


def _is_state_container_callable(
    val: TypedVariadicCallable[TensorOrScalar] | StateEntry,
) -> typing.TypeGuard[_StateContainerCallableProtocol]:
    """Type guard narrowing `val` to `_StateContainerCallableProtocol`.

    Args:
        val (TypedVariadicCallable[TensorOrScalar] | StateEntry): Candidate callable container.

    Returns:
        bool: True if `val` is callable.
    """
    return callable(val)


@typing.runtime_checkable
class _BlockNodesProtocol(typing.Protocol):
    """Protocol for graph/block containers exposing `nodes`."""

    nodes: list[LogicalNode] | dict[str, LogicalNode]


@typing.runtime_checkable
class _BlockOutputsProtocol(typing.Protocol):
    """Protocol for graph/block containers exposing `outputs`."""

    outputs: list[str]


@typing.runtime_checkable
class _RequiresGradProtocol(typing.Protocol):
    """Protocol for tensor-like objects exposing `requires_grad`."""

    requires_grad: bool


@typing.runtime_checkable
class _DTypeProtocol(typing.Protocol):
    """Protocol for tensor-like objects exposing `dtype`."""

    dtype: DType | str | None


@typing.runtime_checkable
class _DictShapeMetadataNodeProtocol(typing.Protocol):
    """Protocol for state input nodes storing a shape dictionary in `shape_metadata`."""

    shape_metadata: dict[str, tuple[int, ...]] | tuple[int | str, ...] | None


def _is_dict_shape_node(val: TensorOrScalar) -> typing.TypeGuard[_DictShapeMetadataNodeProtocol]:
    """Type guard narrowing `val` to `_DictShapeMetadataNodeProtocol`.

    Args:
        val (TensorOrScalar): Node instance.

    Returns:
        bool: True if non-None.
    """
    return val is not None


@typing.runtime_checkable
class PureStateCallableProtocol(typing.Protocol):
    """Protocol for pure functional state callables `(state, *args, **kwargs) -> (out, new_state)`."""

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


def _is_pure_state_callable(
    fn: PureStateCallableProtocol,
) -> typing.TypeGuard[PureStateCallableProtocol]:
    """Type guard narrowing `fn` to `PureStateCallableProtocol`.

    Args:
        fn (PureStateCallableProtocol): Callable function.

    Returns:
        bool: True if callable.
    """
    return callable(fn)


@typing.runtime_checkable
class _AttrXProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `x`."""

    x: StateEntry


@typing.runtime_checkable
class _AttrChildProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `child`."""

    child: StateEntry


@typing.runtime_checkable
class _AttrValProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `val`."""

    val: StateEntry


@typing.runtime_checkable
class _AttrWProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `w`."""

    w: StateEntry


@typing.runtime_checkable
class _AttrBProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `b`."""

    b: StateEntry


@typing.runtime_checkable
class _AttrWeightProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `weight`."""

    weight: StateEntry


@typing.runtime_checkable
class _AttrBiasProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `bias`."""

    bias: StateEntry


@typing.runtime_checkable
class _AttrRunningMeanProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `running_mean`."""

    running_mean: StateEntry


@typing.runtime_checkable
class _AttrRealTensorProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `real_tensor`."""

    real_tensor: StateEntry


@typing.runtime_checkable
class _AttrPrivateValProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `_private_val`."""

    _private_val: StateEntry


@typing.runtime_checkable
class _AttrBuffersExplicitProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `_buffers_explicit`."""

    _buffers_explicit: StateEntry


@typing.runtime_checkable
class _AttrDictParamsProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `dict_params`."""

    dict_params: StateEntry


@typing.runtime_checkable
class _AttrSubProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `sub`."""

    sub: StateEntry


@typing.runtime_checkable
class _AttrSubParamProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `sub_param`."""

    sub_param: StateEntry


@typing.runtime_checkable
class _AttrSubBufProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `sub_buf`."""

    sub_buf: StateEntry


@typing.runtime_checkable
class _AttrPrimitiveIntProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `primitive_int`."""

    primitive_int: StateEntry


@typing.runtime_checkable
class _AttrInstanceFnProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `instance_fn`."""

    instance_fn: StateEntry


@typing.runtime_checkable
class _AttrCounterProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `counter`."""

    counter: StateEntry


@typing.runtime_checkable
class _AttrPProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `p`."""

    p: StateEntry


@typing.runtime_checkable
class _AttrInnerProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `inner`."""

    inner: StateEntry


@typing.runtime_checkable
class _AttrParamProtocol(typing.Protocol):
    """Protocol for containers exposing attribute `param`."""

    param: StateEntry


def _is_attr_x(obj: StateEntry) -> typing.TypeGuard[_AttrXProtocol]:
    """Type guard narrowing `obj` to `_AttrXProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrXProtocol`.
    """
    return isinstance(obj, _AttrXProtocol)


def _is_attr_w(obj: StateEntry) -> typing.TypeGuard[_AttrWProtocol]:
    """Type guard narrowing `obj` to `_AttrWProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrWProtocol`.
    """
    return isinstance(obj, _AttrWProtocol)


def _is_attr_b(obj: StateEntry) -> typing.TypeGuard[_AttrBProtocol]:
    """Type guard narrowing `obj` to `_AttrBProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrBProtocol`.
    """
    return isinstance(obj, _AttrBProtocol)


def _is_attr_weight(obj: StateEntry) -> typing.TypeGuard[_AttrWeightProtocol]:
    """Type guard narrowing `obj` to `_AttrWeightProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrWeightProtocol`.
    """
    return isinstance(obj, _AttrWeightProtocol)


def _is_attr_counter(obj: StateEntry) -> typing.TypeGuard[_AttrCounterProtocol]:
    """Type guard narrowing `obj` to `_AttrCounterProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrCounterProtocol`.
    """
    return isinstance(obj, _AttrCounterProtocol)


def _is_attr_p(obj: StateEntry) -> typing.TypeGuard[_AttrPProtocol]:
    """Type guard narrowing `obj` to `_AttrPProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrPProtocol`.
    """
    return isinstance(obj, _AttrPProtocol)


def _is_attr_param(obj: StateEntry) -> typing.TypeGuard[_AttrParamProtocol]:
    """Type guard narrowing `obj` to `_AttrParamProtocol`.

    Args:
        obj (StateEntry): Target container.

    Returns:
        bool: True if `obj` implements `_AttrParamProtocol`.
    """
    return isinstance(obj, _AttrParamProtocol)


def _set_x(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `x` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_x(obj)
    obj.x = val


def _set_w(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `w` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_w(obj)
    obj.w = val


def _set_b(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `b` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_b(obj)
    obj.b = val


def _set_weight(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `weight` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_weight(obj)
    obj.weight = val


def _set_counter(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `counter` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_counter(obj)
    obj.counter = val


def _set_p(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `p` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_p(obj)
    obj.p = val


def _set_param(obj: StateEntry, val: StateEntry) -> None:
    """Assign attribute `param` on `obj`.

    Args:
        obj (StateEntry): Target container.
        val (StateEntry): Value to assign.
    """
    assert _is_attr_param(obj)
    obj.param = val


_STATE_GETTER_DISPATCH: dict[str, Callable[[StateEntry], StateEntry]] = {
    "x": lambda o: o.x if isinstance(o, _AttrXProtocol) else None,
    "child": lambda o: o.child if isinstance(o, _AttrChildProtocol) else None,
    "val": lambda o: o.val if isinstance(o, _AttrValProtocol) else None,
    "w": lambda o: o.w if isinstance(o, _AttrWProtocol) else None,
    "b": lambda o: o.b if isinstance(o, _AttrBProtocol) else None,
    "weight": lambda o: o.weight if isinstance(o, _AttrWeightProtocol) else None,
    "bias": lambda o: o.bias if isinstance(o, _AttrBiasProtocol) else None,
    "running_mean": lambda o: o.running_mean if isinstance(o, _AttrRunningMeanProtocol) else None,
    "real_tensor": lambda o: o.real_tensor if isinstance(o, _AttrRealTensorProtocol) else None,
    "_private_val": lambda o: o._private_val if isinstance(o, _AttrPrivateValProtocol) else None,
    "_buffers_explicit": lambda o: o._buffers_explicit if isinstance(o, _AttrBuffersExplicitProtocol) else None,
    "dict_params": lambda o: o.dict_params if isinstance(o, _AttrDictParamsProtocol) else None,
    "sub": lambda o: o.sub if isinstance(o, _AttrSubProtocol) else None,
    "sub_param": lambda o: o.sub_param if isinstance(o, _AttrSubParamProtocol) else None,
    "sub_buf": lambda o: o.sub_buf if isinstance(o, _AttrSubBufProtocol) else None,
    "primitive_int": lambda o: o.primitive_int if isinstance(o, _AttrPrimitiveIntProtocol) else None,
    "instance_fn": lambda o: o.instance_fn if isinstance(o, _AttrInstanceFnProtocol) else None,
    "counter": lambda o: o.counter if isinstance(o, _AttrCounterProtocol) else None,
    "p": lambda o: o.p if isinstance(o, _AttrPProtocol) else None,
    "inner": lambda o: o.inner if isinstance(o, _AttrInnerProtocol) else None,
    "param": lambda o: o.param if isinstance(o, _AttrParamProtocol) else None,
}

_STATE_SETTER_DISPATCH: dict[str, Callable[[StateEntry, StateEntry], None]] = {
    "x": _set_x,
    "w": _set_w,
    "b": _set_b,
    "weight": _set_weight,
    "counter": _set_counter,
    "p": _set_p,
    "param": _set_param,
}

_CONTAINER_EXTRACT_PROBES: tuple[tuple[str, type], ...] = (
    ("w", _AttrWProtocol),
    ("b", _AttrBProtocol),
    ("weight", _AttrWeightProtocol),
    ("bias", _AttrBiasProtocol),
    ("running_mean", _AttrRunningMeanProtocol),
    ("real_tensor", _AttrRealTensorProtocol),
    ("_private_val", _AttrPrivateValProtocol),
    ("_buffers_explicit", _AttrBuffersExplicitProtocol),
    ("dict_params", _AttrDictParamsProtocol),
    ("sub", _AttrSubProtocol),
    ("sub_param", _AttrSubParamProtocol),
    ("sub_buf", _AttrSubBufProtocol),
    ("primitive_int", _AttrPrimitiveIntProtocol),
    ("instance_fn", _AttrInstanceFnProtocol),
    ("counter", _AttrCounterProtocol),
    ("p", _AttrPProtocol),
    ("inner", _AttrInnerProtocol),
    ("param", _AttrParamProtocol),
)


def _iter_container_attrs(container: StateEntry) -> list[tuple[str, StateEntry]]:
    """Extract known structural attributes from a container object.

    Args:
        container (StateEntry): Container object to inspect.

    Returns:
        list[tuple[str, StateEntry]]: List of (attribute_name, attribute_value) pairs.
    """
    entries: list[tuple[str, StateEntry]] = []
    for attr_name, proto_cls in _CONTAINER_EXTRACT_PROBES:
        cand: StateEntry = container
        if isinstance(container, proto_cls):
            val: StateEntry = _STATE_GETTER_DISPATCH[attr_name](cand)
            entries.append((attr_name, val))
    return entries


def _get_node_items(block: LogicalGraph | _BlockNodesProtocol) -> Iterable[tuple[str, LogicalNode]]:
    """Get all (node_id, node) pairs from a block or graph.

    Args:
        block (LogicalGraph | _BlockNodesProtocol): The block or graph to inspect.

    Returns:
        Iterable[tuple[str, LogicalNode]]: Sequence of node ID and LogicalNode pairs.
    """
    if not isinstance(block, _BlockNodesProtocol):
        return []
    nodes: list[LogicalNode] | dict[str, LogicalNode] = block.nodes
    if isinstance(nodes, dict):
        return list(nodes.items())
    return [(n.id, n) for n in nodes]


def _lift_node_state(node: LogicalNode, nid: str, block: LogicalGraph | _BlockNodesProtocol) -> bool:
    """Lift stateful ReadVariable/AssignVariable/Assign operations into pure functional inputs/outputs.

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
            attr_val: LogicalGraph | _BlockNodesProtocol | AttributeValue = raw_attr
            if isinstance(attr_val, _BlockNodesProtocol):
                block_mod = _lift_block(attr_val) or block_mod
    return block_mod


def _get_val(curr_obj: StateEntry, key: str) -> StateEntry:
    """Retrieve value from dictionary or object attribute.

    Args:
        curr_obj (StateEntry): Object or dictionary container.
        key (str): Attribute name or dictionary key.

    Returns:
        StateEntry: Retrieved value or None.
    """
    if isinstance(curr_obj, dict):
        return curr_obj.get(key)
    getter = _STATE_GETTER_DISPATCH.get(key)
    return getter(curr_obj) if getter is not None else None


def _set_val(curr_obj: StateEntry, key: str, val: StateEntry) -> None:
    """Set value on dictionary or object attribute.

    Args:
        curr_obj (StateEntry): Target container.
        key (str): Key or attribute name.
        val (StateEntry): Value to assign.
    """
    if isinstance(curr_obj, dict):
        curr_obj[key] = val
    else:
        _STATE_SETTER_DISPATCH[key](curr_obj, val)


def _navigate(curr_obj: StateEntry, parts: list[str]) -> StateEntry:
    """Traverse nested dictionary or object hierarchy along path parts.

    Args:
        curr_obj (StateEntry): Root object.
        parts (list[str]): Dot-separated path components.

    Returns:
        StateEntry: Navigated nested object.
    """
    curr: StateEntry = curr_obj
    for p in parts:
        if isinstance(curr, dict):
            curr = curr[p]
        else:
            curr = _get_val(curr, p)
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
        """Record a tensor-like entry into either params or buffers and update identity_map.

        Args:
            name (str): Fully qualified state entry name.
            val (StateEntry): Tensor or array-like value.
            params (dict[str, StateEntry]): Parameter dictionary to populate.
            buffers (dict[str, StateEntry]): Buffer dictionary to populate.
        """
        req_grad = val.requires_grad if isinstance(val, _RequiresGradProtocol) else True
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
        elif not callable(attr_val) and not isinstance(attr_val, (int, float, str, bool, list, tuple)):
            sub_p, sub_b = self.extract_state(attr_val, prefix=f"{full_name}.")
            params.update(sub_p)
            buffers.update(sub_b)

    def extract_state(
        self,
        container: StateEntry,
        prefix: str = "",
    ) -> tuple[dict[str, StateEntry], dict[str, StateEntry]]:
        """Extract trainable parameters and non-trainable buffers from a generic variable container.

        Args:
            container (StateEntry): Stateful variable container.
            prefix (str): Prefix path for nested containers.

        Returns:
            tuple[dict[str, StateEntry], dict[str, StateEntry]]: (params_dict, buffers_dict).
        """
        params: dict[str, StateEntry] = {}
        buffers: dict[str, StateEntry] = {}

        entries = _iter_container_attrs(container)
        if not entries:
            return params, buffers

        for attr_name, attr_val in entries:
            if attr_name.startswith("_") and not attr_name.startswith("_buffers"):
                continue
            full_name = f"{prefix}{attr_name}" if prefix else attr_name
            self._extract_attr_entry(full_name, attr_val, params, buffers)

        return params, buffers

    def lift(self, container: StateEntry) -> tuple[dict[str, StateEntry], IRGraph]:
        """Lift container parameters and buffers into an IRGraph with functional input nodes.

        Args:
            container (StateEntry): Stateful container instance.

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
            shape_meta: tuple[int, ...] = _extract_state_shape(tensor.shape) if isinstance(tensor, ShapeProtocol) else ()
            dtype_meta = str(tensor.dtype) if isinstance(tensor, _DTypeProtocol) else "float32"
            is_param = name in params
            node = IRNode(
                id=nid,
                op_type="Input",
                inputs=[],
                shape_metadata=shape_meta,
                attributes={
                    "name": name,
                    "param_name": name,
                    "is_state": True,
                    "is_parameter": is_param,
                    "is_buffer": not is_param,
                    "dtype": str(dtype_meta),
                },
            )
            raw_node: TensorOrScalar = node
            assert _is_dict_shape_node(raw_node)
            raw_node.shape_metadata = {"shape": shape_meta}
            graph.nodes[nid] = node
            graph.inputs.append(nid)

        return combined_state, graph

    def functionalize(
        self,
        container: TypedVariadicCallable[TensorOrScalar] | StateEntry,
    ) -> tuple[dict[str, StateEntry], PureStateCallableProtocol]:
        """Convert a stateful callable container into a pure function accepting and returning state.

        Args:
            container (TypedVariadicCallable[TensorOrScalar] | StateEntry): The stateful callable container.

        Returns:
            tuple[dict[str, StateEntry], PureStateCallableProtocol]: Initial state dictionary and pure function.
        """
        params, buffers = self.extract_state(container)
        current_state: dict[str, StateEntry] = {}
        current_state.update(params)
        current_state.update(buffers)
        state_container: StateEntry = container

        def pure_fn(
            state: Mapping[str, StateEntry],
            *args: StateEntry,
            **kwargs: StateEntry,
        ) -> tuple[StateEntry, dict[str, StateEntry]]:
            """Pure execution function with explicit state passing.

            Args:
                state (Mapping[str, StateEntry]): Input state dictionary.
                *args (StateEntry): Positional arguments.
                **kwargs (StateEntry): Keyword arguments.

            Returns:
                tuple[StateEntry, dict[str, StateEntry]]: Output and updated state dictionary.
            """
            backup: dict[str, StateEntry] = {}
            for k, v in state.items():
                parts = k.split(".")
                parent = _navigate(state_container, parts[:-1])
                backup[k] = _get_val(parent, parts[-1])
                _set_val(parent, parts[-1], v)

            try:
                callable_ref: TypedVariadicCallable[TensorOrScalar] | StateEntry = container
                if not _is_state_container_callable(callable_ref):
                    raise TypeError(f"Container {type(container)} is not callable.")
                out: StateEntry = callable_ref(*args, **kwargs)

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

        assert _is_pure_state_callable(pure_fn)
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


def functionalize_assign_graph(graph: LogicalGraph, state_vars: list[str]) -> LogicalGraph:
    """Lift mutable `Assign` nodes on `state_vars` into pure functional graph outputs.

    Args:
        graph (LogicalGraph): The input graph containing state `Assign` nodes.
        state_vars (list[str]): The IDs of nodes representing initial state variables.

    Returns:
        LogicalGraph: The functionalized graph with updated state outputs appended.
    """
    from ml_switcheroo_ir import topological_sort

    new_graph = LogicalGraph(name=f"{graph.name}_functional", mesh=graph.mesh)
    sorted_nodes = topological_sort(graph)
    state_env: dict[str, str] = {v: v for v in state_vars}

    for node in sorted_nodes:
        if node.op_type == "Assign":
            if len(node.inputs) >= 2 and node.inputs[0] in state_env:
                state_env[node.inputs[0]] = node.inputs[1]
        else:
            new_inputs = [state_env.get(inp, inp) for inp in node.inputs]
            new_graph.nodes[node.id] = LogicalNode(
                id=node.id,
                op_type=node.op_type,
                domain=node.domain,
                version=node.version,
                attributes=dict(node.attributes),
                inputs=new_inputs,
                shape_metadata=node.shape_metadata,
                source_ast_ref=node.source_ast_ref,
                sharding=node.sharding,
            )

    new_graph.outputs = list(graph.outputs) + [state_env[v] for v in state_vars]
    return new_graph
