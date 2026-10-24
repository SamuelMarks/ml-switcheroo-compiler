"""SPMD compiler pass."""

from __future__ import annotations

import builtins  # noqa: F401
import os
import typing
from pathlib import Path

import pydantic
import yaml
from ml_switcheroo_ir import AttributeValue, PartitionSpec

from ml_switcheroo_compiler.backends.registry import as_int
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, as_opt_pspec

ShardingValue = typing.Union["SPMDShardingAnnotation", PartitionSpec, TensorOrScalar]


def _get_sharding_axes(sharding: ShardingValue) -> list[str]:
    """Extract non-None mesh axis names from `sharding`.

    Args:
        sharding (ShardingValue): The sharding parameter.

    Returns:
        list[str]: Active sharding axis names.
    """
    if not sharding or not hasattr(sharding, "mesh_mapping"):
        return []
    return [str(m) for m in getattr(sharding, "mesh_mapping") if m is not None]


def _is_boundary_transition(inp_sharding: ShardingValue, node_sharding: ShardingValue) -> tuple[bool, bool]:
    """Check whether `inp_sharding` and `node_sharding` are sharded.

    Args:
        inp_sharding (ShardingValue): The inp_sharding parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        tuple[bool, bool]: `(inp_sharded, node_sharded)` boolean flags.
    """
    inp_sharded = bool(_get_sharding_axes(inp_sharding))
    node_sharded = bool(_get_sharding_axes(node_sharding))
    return inp_sharded, node_sharded


def _create_all_gather_node(inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Create an `AllGather` collective communication node.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Constructed `AllGather` node.
    """
    return IRNode(id=f"{inp_id}_all_gather", op_type="AllGather", inputs=[inp_id], sharding=as_opt_pspec(node_sharding), attributes={"dispatch_early": True})


def _create_reduce_scatter_node(inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Create a `ReduceScatter` collective communication node.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Constructed `ReduceScatter` node.
    """
    return IRNode(id=f"{inp_id}_reduce_scatter", op_type="ReduceScatter", inputs=[inp_id], sharding=as_opt_pspec(node_sharding), attributes={"dispatch_early": True})


def _create_all_reduce_node(inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Create an `AllReduce` collective communication node.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Constructed `AllReduce` node.
    """
    return IRNode(id=f"{inp_id}_all_reduce", op_type="AllReduce", inputs=[inp_id], sharding=as_opt_pspec(node_sharding), attributes={"dispatch_early": True})


def _create_all_to_all_node(inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Create an `AllToAll` collective communication node.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Constructed `AllToAll` node.
    """
    return IRNode(id=f"{inp_id}_all_to_all", op_type="AllToAll", inputs=[inp_id], sharding=as_opt_pspec(node_sharding), attributes={"dispatch_early": True})


def _inject_all_gather(node: IRNode, idx: int, inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Inject an `AllGather` node before `node.inputs[idx]`.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Injected `AllGather` node.
    """
    gather_node = _create_all_gather_node(inp_id, node_sharding)
    node.inputs[idx] = gather_node.id
    return gather_node


def _inject_reduce_scatter(node: IRNode, idx: int, inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Inject a `ReduceScatter` node before `node.inputs[idx]`.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Injected `ReduceScatter` node.
    """
    scatter_node = _create_reduce_scatter_node(inp_id, node_sharding)
    node.inputs[idx] = scatter_node.id
    return scatter_node


def _inject_all_reduce(node: IRNode, idx: int, inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Inject an `AllReduce` node before `node.inputs[idx]`.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Injected `AllReduce` node.
    """
    reduce_node = _create_all_reduce_node(inp_id, node_sharding)
    node.inputs[idx] = reduce_node.id
    return reduce_node


def _inject_all_to_all(node: IRNode, idx: int, inp_id: str, node_sharding: ShardingValue) -> IRNode:
    """Inject an `AllToAll` node before `node.inputs[idx]`.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode: Injected `AllToAll` node.
    """
    atoa_node = _create_all_to_all_node(inp_id, node_sharding)
    node.inputs[idx] = atoa_node.id
    return atoa_node


_SPMD_RULES: dict[str, AttributeValue] | None = None


def _get_spmd_rules() -> dict[str, AttributeValue]:
    """Load and cache SPMD communication and propagation rules from YAML.

    Returns:
        dict[str, AttributeValue]: Loaded SPMD rules dictionary.
    """
    global _SPMD_RULES
    if _SPMD_RULES is None:
        _SPMD_RULES = {}
        yaml_dir = Path(__file__).parent / "spmd_mappings"
        if yaml_dir.exists():
            for filename in os.listdir(yaml_dir):
                if filename.endswith(".yaml"):
                    with open(yaml_dir / filename) as f:
                        op_data = yaml.safe_load(f)
                    if isinstance(op_data, dict):
                        _SPMD_RULES.update(op_data)
        else:
            yaml_path = Path(__file__).parent / "spmd_mappings.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    _SPMD_RULES = yaml.safe_load(f) or {}
    return _SPMD_RULES


def _matches_spmd_condition(
    cond: dict[str, bool | str],
    is_reduction: bool,
    is_grad: bool,
    inp_axes: list[str],
    node_axes: list[str],
) -> bool:
    """Evaluate whether a single SPMD communication condition entry matches the current transition.

    Args:
        cond (dict[str, bool | str]): Condition dictionary from communication_matrix.
        is_reduction (bool): True if the consumer node is a reduction operation.
        is_grad (bool): True if the consumer node is a gradient/adjoint operation.
        inp_axes (list[str]): Sharding axes of the input tensor.
        node_axes (list[str]): Sharding axes of the target node.

    Returns:
        bool: True if the condition matches, False otherwise.
    """
    if cond.get("default", False):
        return True
    if cond.get("is_reduction", False) and is_reduction:
        return True
    if cond.get("is_grad", False) and is_grad:
        return True
    if cond.get("axes_match") is False and cond.get("axes_length_match") is True:
        return inp_axes != node_axes and len(inp_axes) == len(node_axes)
    return False


def _determine_spmd_communication(
    node: IRNode,
    idx: int,
    inp_id: str,
    node_sharding: SPMDShardingAnnotation | ShardingValue | None,
    inp_axes: list[str],
    node_axes: list[str],
) -> IRNode | None:
    """Determine the type of SPMD communication needed using data-driven rules.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (SPMDShardingAnnotation | None): The node_sharding parameter.
        inp_axes (list[str]): The inp_axes parameter.
        node_axes (list[str]): The node_axes parameter.

    Returns:
        IRNode | None: Injected collective communication node or None.
    """
    rules = _get_spmd_rules()
    state = [bool(inp_axes), bool(node_axes)]
    raw_reds = rules.get("reductions", [])
    reductions: list[str] = [str(r) for r in raw_reds] if isinstance(raw_reds, (list, tuple)) else []
    is_reduction = getattr(node, "op_type", "") in reductions
    node_id_str = getattr(node, "id", "")
    is_grad = "grad" in node_id_str or "adjoint" in node_id_str

    injected_op = "none"
    raw_comm = rules.get("communication_matrix", [])
    comm_matrix: list[dict[str, AttributeValue]] = [dict(r) for r in raw_comm if isinstance(r, dict)] if isinstance(raw_comm, list) else []
    for rule in comm_matrix:
        if rule.get("state") == state:
            raw_conds = rule.get("conditions", [])
            conditions: list[dict[str, bool | str]] = [{str(k): v for k, v in c.items() if isinstance(v, (bool, str))} for c in raw_conds if isinstance(c, dict)] if isinstance(raw_conds, list) else []
            for cond in conditions:
                if _matches_spmd_condition(cond, is_reduction, is_grad, inp_axes, node_axes):
                    injected_op = str(cond.get("inject", "none"))
                    break
            break

    injectors = {
        "AllReduce": _inject_all_reduce,
        "AllGather": _inject_all_gather,
        "ReduceScatter": _inject_reduce_scatter,
        "AllToAll": _inject_all_to_all,
    }
    injector = injectors.get(injected_op)
    return injector(node, idx, inp_id, node_sharding) if injector is not None else None


def _process_spmd_input(node: IRNode, idx: int, inp_id: str, graph: IRGraph, node_sharding: ShardingValue) -> IRNode | None:
    """Inspect a single node input and inject collective communication if sharding transitions require it.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        graph (IRGraph): The graph parameter.
        node_sharding (ShardingValue): The node_sharding parameter.

    Returns:
        IRNode | None: Injected collective node or None.
    """
    if inp_id not in graph.nodes:
        return None

    inp_node = graph.nodes[inp_id]
    inp_sharding = getattr(inp_node, "sharding", None)

    if not inp_sharding:
        return None

    inp_axes = _get_sharding_axes(inp_sharding)
    node_axes = _get_sharding_axes(node_sharding)

    return _determine_spmd_communication(
        node,
        idx,
        inp_id,
        node_sharding if isinstance(node_sharding, SPMDShardingAnnotation) else None,
        inp_axes,
        node_axes,
    )


def _process_spmd_node(node: IRNode, graph: IRGraph) -> tuple[bool, list[IRNode]]:
    """Process all inputs of `node` for SPMD collective injection.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.

    Returns:
        tuple[bool, list[IRNode]]: `(modified, injected_nodes)`.
    """
    modified = False
    injected_nodes = []

    node_sharding = getattr(node, "sharding", None)
    if not node_sharding:
        return False, []

    for idx, inp_id in enumerate(list(node.inputs)):
        inj_node = _process_spmd_input(node, idx, inp_id, graph, node_sharding)
        if inj_node:
            injected_nodes.append(inj_node)
            modified = True

    return modified, injected_nodes


def inject_spmd_communication_pass(graph: IRGraph) -> bool:
    """Injects all_gather, reduce_scatter, all_reduce for SPMD execution.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: True if graph was modified.
    """
    modified = False
    new_nodes = {}

    for node_id, node in list(graph.nodes.items()):
        new_nodes[node_id] = node

        node_modified, injected = _process_spmd_node(node, graph)
        if node_modified:
            modified = True

        for inj_node in injected:
            new_nodes[inj_node.id] = inj_node

    from ml_switcheroo_ir import NodeDict

    graph.nodes = NodeDict(new_nodes)
    return modified


class SPMDShardingAnnotation(PartitionSpec):
    """Lightweight multi-axis sharding specification container."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mesh: AttributeValue = None
    mesh_mapping: tuple[str | None, ...] = ()
    mesh_shape: tuple[int, ...] | None = None
    mesh_axes: tuple[str, ...] | None = None

    def __init__(
        self,
        mesh: AttributeValue,
        mesh_mapping: typing.Sequence[str | None],
        mesh_shape: typing.Sequence[int] | None = None,
        mesh_axes: typing.Sequence[str] | None = None,
    ) -> None:
        """Initialize SPMDShardingAnnotation.

        Args:
            mesh (AttributeValue): DeviceMesh or mesh configuration.
            mesh_mapping (typing.Sequence[str | None]): Axis sharding names.
            mesh_shape (typing.Sequence[int] | None): Optional device mesh dimensions.
            mesh_axes (typing.Sequence[str] | None): Optional device mesh axis names (e.g., ['dp', 'tp', 'pp']).
        """
        super().__init__(axes=tuple(mesh_mapping))
        self.mesh = mesh
        self.mesh_mapping = tuple(mesh_mapping)
        self.mesh_shape = tuple(mesh_shape) if mesh_shape is not None else None
        self.mesh_axes = tuple(mesh_axes) if mesh_axes is not None else None

    @property
    def is_sharded(self) -> bool:
        """Check if any tensor dimension is sharded across mesh axes.

        Returns:
            bool: True if at least one axis is sharded.
        """
        return any(m is not None for m in self.mesh_mapping)

    def __repr__(self) -> str:
        """Return representation.

        Returns:
            str: String representation.
        """
        return f"SPMDShardingAnnotation(mesh={self.mesh}, mesh_mapping={self.mesh_mapping})"


def _set_collective_attribute(node: IRNode) -> None:
    """Set the collective communication attribute on a sharded reduction/contracting node.

    Args:
        node (IRNode): Node requiring collective reduction or scatter.
    """
    if node.attributes.get("requires_scatter") or node.attributes.get("requires_reduce_scatter"):
        node.attributes["inject_collective"] = "ReduceScatter"
    else:
        node.attributes["inject_collective"] = "AllReduce"


def _propagate_elementwise_sharding(node: IRNode, graph: IRGraph) -> bool:
    """Propagate sharding from the first sharded input to an elementwise or convolution node.

    Args:
        node (IRNode): Target node.
        graph (IRGraph): Computation graph.

    Returns:
        bool: True if sharding was propagated, False otherwise.
    """
    for inp_id in node.inputs:
        inp_node = graph.nodes.get(inp_id)
        if inp_node and getattr(inp_node, "sharding", None) is not None:
            node.sharding = inp_node.sharding
            return True
    return False


def _propagate_matmul_sharding(node: IRNode, graph: IRGraph) -> bool:
    """Propagate sharding across matrix multiplication operations.

    Args:
        node (IRNode): MatMul/Linear IR node.
        graph (IRGraph): Computation graph.

    Returns:
        bool: True if sharding was propagated, False otherwise.
    """
    lhs = graph.nodes.get(node.inputs[0])
    rhs = graph.nodes.get(node.inputs[1])
    lhs_sharding = getattr(lhs, "sharding", None) if lhs else None
    rhs_sharding = getattr(rhs, "sharding", None) if rhs else None

    lhs_map: list[str | None] = list(getattr(lhs_sharding, "mesh_mapping", []))
    rhs_map: list[str | None] = list(getattr(rhs_sharding, "mesh_mapping", []))
    if not lhs_map and not rhs_map:
        return False

    mesh = getattr(lhs_sharding, "mesh", None) or getattr(rhs_sharding, "mesh", None)
    while len(lhs_map) < 2:
        lhs_map.insert(0, None)
    while len(rhs_map) < 2:
        rhs_map.append(None)

    lhs_contracting = lhs_map[-1]
    rhs_contracting = rhs_map[-2]
    out_mapping = list(lhs_map[:-1]) + [rhs_map[-1]]

    if lhs_contracting is not None and lhs_contracting == rhs_contracting:
        cleaned_mapping = [m if m != lhs_contracting else None for m in out_mapping]
        node.sharding = SPMDShardingAnnotation(mesh, cleaned_mapping)
        _set_collective_attribute(node)
    elif (lhs_map[-2] is not None and rhs_contracting is None) or (rhs_map[-1] is not None and lhs_contracting is None):
        node.sharding = SPMDShardingAnnotation(mesh, out_mapping)
    else:
        node.sharding = lhs_sharding or rhs_sharding
    return True


def _propagate_reduction_sharding(node: IRNode, graph: IRGraph) -> bool:
    """Propagate sharding across reduction operations.

    Args:
        node (IRNode): Reduction IR node.
        graph (IRGraph): Computation graph.

    Returns:
        bool: True if sharding was propagated, False otherwise.
    """
    inp = graph.nodes.get(node.inputs[0])
    inp_sharding = getattr(inp, "sharding", None) if inp else None
    if inp_sharding is None:
        return False

    inp_map: list[str | None] = list(getattr(inp_sharding, "mesh_mapping", []))
    mesh = getattr(inp_sharding, "mesh", None)
    reduce_axis = node.attributes.get("axis", node.attributes.get("axes", 0))
    raw_axes = [int(as_int(a)) for a in reduce_axis] if isinstance(reduce_axis, (list, tuple)) else [int(as_int(reduce_axis))]
    norm_reduce_axes = {a if a >= 0 else a + len(inp_map) for a in raw_axes}

    is_reduced_dim_sharded = False
    out_map: list[str | None] = []
    for dim_idx, m in enumerate(inp_map):
        if dim_idx in norm_reduce_axes:
            if m is not None:
                is_reduced_dim_sharded = True
        else:
            out_map.append(m)

    node.sharding = SPMDShardingAnnotation(mesh, out_map or [None])
    if is_reduced_dim_sharded:
        _set_collective_attribute(node)
    return True


def propagate_sharding(graph: IRGraph) -> bool:
    """Propagate sharding annotations across matmul, reduction, and elementwise nodes.

    Args:
        graph (IRGraph): Target computation graph.

    Returns:
        bool: True if any node sharding was updated.
    """
    rules = _get_spmd_rules()
    prop_rules = rules.get("propagation_rules", {})
    elementwise_ops = set(getattr(getattr(prop_rules, "get")("elementwise", {}), "get")("ops", [])) | {"Add", "Sub", "Mul", "Div", "Neg", "Relu", "GELU", "Sigmoid", "Tanh"}
    matmul_ops = set(getattr(getattr(prop_rules, "get")("matmul", {}), "get")("ops", ["MatMul", "BatchMatMul"])) | {"MatMul", "BatchMatMul", "Dot", "Linear"}
    reduction_ops = set(getattr(getattr(prop_rules, "get")("reductions", {}), "get")("ops", [])) | set(getattr(rules, "get")("reductions", [])) | {"Sum"}
    conv_ops = set(getattr(getattr(prop_rules, "get")("spatial_conv", {}), "get")("ops", ["Conv1D", "Conv2D", "Conv3D"]))

    modified = False
    for node in graph.nodes.values():
        if getattr(node, "sharding", None) is not None:
            continue

        op_type = getattr(node, "op_type", "")
        if op_type in elementwise_ops:
            modified = _propagate_elementwise_sharding(node, graph) or modified
        elif op_type in matmul_ops and len(node.inputs) >= 2:
            modified = _propagate_matmul_sharding(node, graph) or modified
        elif (op_type in reduction_ops or op_type.startswith("Reduce")) and len(node.inputs) >= 1:
            modified = _propagate_reduction_sharding(node, graph) or modified
        elif op_type in conv_ops and len(node.inputs) >= 1:
            modified = _propagate_elementwise_sharding(node, graph) or modified

    return modified


def spmd_partitioning_pass(graph: IRGraph) -> bool:
    """Execute complete SPMD partitioning: sharding propagation and communication injection.

    Args:
        graph (IRGraph): The IR graph to partition.

    Returns:
        bool: True if the graph was modified.
    """
    prop_modified = propagate_sharding(graph)
    comm_modified = inject_spmd_communication_pass(graph)

    collective_injected = False
    for node_id, node in list(graph.nodes.items()):
        coll_type = node.attributes.get("inject_collective")
        if coll_type in ("AllReduce", "ReduceScatter", "AllGather", "AllToAll"):
            node.attributes.pop("inject_collective", None)
            coll_id = f"{node_id}_all_reduce" if coll_type == "AllReduce" else f"{node_id}_{str(coll_type).lower()}"
            if coll_id not in graph.nodes:
                if coll_type == "AllReduce":
                    coll_node = _create_all_reduce_node(node_id, getattr(node, "sharding", None))
                elif coll_type == "ReduceScatter":
                    coll_node = _create_reduce_scatter_node(node_id, getattr(node, "sharding", None))
                elif coll_type == "AllGather":
                    coll_node = _create_all_gather_node(node_id, getattr(node, "sharding", None))
                else:
                    coll_node = _create_all_to_all_node(node_id, getattr(node, "sharding", None))
                for other_node in graph.nodes.values():
                    if other_node.id != coll_id and node_id in other_node.inputs:
                        other_node.inputs = [coll_id if inp == node_id else inp for inp in other_node.inputs]
                if hasattr(graph, "outputs") and node_id in graph.outputs:
                    graph.outputs = [coll_id if out == node_id else out for out in graph.outputs]
                graph.nodes[coll_id] = coll_node
                collective_injected = True

    return prop_modified or comm_modified or collective_injected
