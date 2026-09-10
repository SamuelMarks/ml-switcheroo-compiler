"""Module spmd.py."""

from __future__ import annotations

import os
import typing

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""SPMD compiler pass."""


from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _get_sharding_axes(sharding) -> list[str]:
    """Evaluate _get_sharding_axes operation.

    Args:
        sharding (typing.Union[dict, list, tuple, None]): The sharding parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if not sharding or not hasattr(sharding, "mesh_mapping"):
        return []
    return [m for m in sharding.mesh_mapping if m is not None]


def _is_boundary_transition(inp_sharding, node_sharding) -> tuple[bool, bool]:
    """Evaluate _is_boundary_transition operation.

    Args:
        inp_sharding (typing.Union[dict, list, tuple, None]): The inp_sharding parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    inp_sharded = bool(_get_sharding_axes(inp_sharding))
    node_sharded = bool(_get_sharding_axes(node_sharding))
    return inp_sharded, node_sharded


def _create_all_gather_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_all_gather_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_gather", op_type="AllGather", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_reduce_scatter_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_reduce_scatter_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_reduce_scatter", op_type="ReduceScatter", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_reduce_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_all_reduce_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_reduce", op_type="AllReduce", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_to_all_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_all_to_all_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_to_all", op_type="AllToAll", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _inject_all_gather(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
    """Evaluate _inject_all_gather operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    gather_node = _create_all_gather_node(inp_id, node_sharding)
    node.inputs[idx] = gather_node.id
    return gather_node


def _inject_reduce_scatter(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
    """Evaluate _inject_reduce_scatter operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    scatter_node = _create_reduce_scatter_node(inp_id, node_sharding)
    node.inputs[idx] = scatter_node.id
    return scatter_node


def _inject_all_reduce(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
    """Evaluate _inject_all_reduce operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    reduce_node = _create_all_reduce_node(inp_id, node_sharding)
    node.inputs[idx] = reduce_node.id
    return reduce_node


def _inject_all_to_all(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
    """Evaluate _inject_all_to_all operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    atoa_node = _create_all_to_all_node(inp_id, node_sharding)
    node.inputs[idx] = atoa_node.id
    return atoa_node


from pathlib import Path

import yaml

_SPMD_RULES = None


def _get_spmd_rules():
    """_get_spmd_rules function.

    Returns:
        dict: Result.
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


def _determine_spmd_communication(
    node: IRNode,
    idx: int,
    inp_id: str,
    node_sharding,
    inp_axes,
    node_axes,
) -> IRNode | None:
    """Determine the type of SPMD communication needed using data-driven rules."""
    rules = _get_spmd_rules()

    inp_sharded = bool(inp_axes)
    node_sharded = bool(node_axes)
    state = [inp_sharded, node_sharded]

    is_reduction = getattr(node, "op_type", "") in rules.get("reductions", [])
    is_grad = "grad" in getattr(node, "id", "") or "adjoint" in getattr(node, "id", "")

    injected_op = "none"

    for rule in rules.get("communication_matrix", []):
        if rule["state"] == state:
            for cond in rule.get("conditions", []):
                if cond.get("default", False):
                    injected_op = cond.get("inject", "none")
                    break
                if cond.get("is_reduction", False) and is_reduction:
                    injected_op = cond.get("inject", "none")
                    break
                if cond.get("is_grad", False) and is_grad:
                    injected_op = cond.get("inject", "none")
                    break
                if cond.get("axes_match") is False and cond.get("axes_length_match") is True:
                    if inp_axes != node_axes and len(inp_axes) == len(node_axes):
                        injected_op = cond.get("inject", "none")
                        break
            break

    if injected_op == "AllReduce":
        return _inject_all_reduce(node, idx, inp_id, node_sharding)
    if injected_op == "AllGather":
        return _inject_all_gather(node, idx, inp_id, node_sharding)
    if injected_op == "ReduceScatter":
        return _inject_reduce_scatter(node, idx, inp_id, node_sharding)
    if injected_op == "AllToAll":
        return _inject_all_to_all(node, idx, inp_id, node_sharding)
    return None


def _process_spmd_input(node: IRNode, idx: int, inp_id: str, graph: IRGraph, node_sharding) -> IRNode | None:
    """Evaluate _process_spmd_input operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        graph (IRGraph): The graph parameter.
        node_sharding (typing.Union[dict, list, tuple, None]): The node_sharding parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if inp_id not in graph.nodes:
        return None

    inp_node = graph.nodes[inp_id]
    inp_sharding = getattr(inp_node, "sharding", None)

    if not inp_sharding:
        return None

    inp_axes = _get_sharding_axes(inp_sharding)
    node_axes = _get_sharding_axes(node_sharding)

    return _determine_spmd_communication(node, idx, inp_id, node_sharding, inp_axes, node_axes)


def _process_spmd_node(node: IRNode, graph: IRGraph) -> tuple[bool, list[IRNode]]:
    """Evaluate _process_spmd_node operation.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.

    Returns:
        tuple: Result.
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
        bool: Result.
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

    graph.nodes = new_nodes
    return modified


class SPMDShardingAnnotation:
    """Lightweight multi-axis sharding specification container."""

    def __init__(
        self,
        mesh: object,
        mesh_mapping: typing.Sequence[str | None],
        mesh_shape: typing.Sequence[int] | None = None,
        mesh_axes: typing.Sequence[str] | None = None,
    ) -> None:
        """Initialize SPMDShardingAnnotation.

        Args:
            mesh (object): DeviceMesh or mesh configuration.
            mesh_mapping (typing.Sequence[str | None]): Axis sharding names.
            mesh_shape (typing.Sequence[int] | None): Optional device mesh dimensions.
            mesh_axes (typing.Sequence[str] | None): Optional device mesh axis names (e.g., ['dp', 'tp', 'pp']).
        """
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


def propagate_sharding(graph: IRGraph) -> bool:
    """Propagate sharding annotations across matmul, reduction, and elementwise nodes.

    Args:
        graph (IRGraph): Target computation graph.

    Returns:
        bool: True if any node sharding was updated.
    """
    rules = _get_spmd_rules()
    prop_rules = rules.get("propagation_rules", {})
    elementwise_ops = set(prop_rules.get("elementwise", {}).get("ops", []))
    matmul_ops = set(prop_rules.get("matmul", {}).get("ops", ["MatMul", "BatchMatMul"]))
    reduction_ops = set(prop_rules.get("reductions", {}).get("ops", [])) | set(rules.get("reductions", []))
    conv_ops = set(prop_rules.get("spatial_conv", {}).get("ops", ["Conv1D", "Conv2D", "Conv3D"]))

    modified = False

    for node in graph.nodes.values():
        if getattr(node, "sharding", None) is not None:
            continue

        op_type = getattr(node, "op_type", "")

        # 1. Elementwise operations: propagate aligned layout
        if op_type in elementwise_ops or op_type in ("Add", "Sub", "Mul", "Div", "Neg", "Relu", "GELU", "Sigmoid", "Tanh"):
            for inp_id in node.inputs:
                inp_node = graph.nodes.get(inp_id)
                if inp_node and getattr(inp_node, "sharding", None) is not None:
                    node.sharding = inp_node.sharding
                    modified = True
                    break

        # 2. Matrix multiplication: row-parallel, col-parallel, contracting-parallel
        elif (op_type in matmul_ops or op_type in ("MatMul", "BatchMatMul", "Dot", "Linear")) and len(node.inputs) >= 2:
            lhs = graph.nodes.get(node.inputs[0])
            rhs = graph.nodes.get(node.inputs[1])
            lhs_sharding = getattr(lhs, "sharding", None) if lhs else None
            rhs_sharding = getattr(rhs, "sharding", None) if rhs else None

            lhs_map = list(getattr(lhs_sharding, "mesh_mapping", []))
            rhs_map = list(getattr(rhs_sharding, "mesh_mapping", []))
            mesh = getattr(lhs_sharding, "mesh", None) or getattr(rhs_sharding, "mesh", None)

            if lhs_map or rhs_map:
                while len(lhs_map) < 2:
                    lhs_map.insert(0, None)
                while len(rhs_map) < 2:
                    rhs_map.append(None)

                lhs_contracting = lhs_map[-1]
                rhs_contracting = rhs_map[-2]

                if lhs_contracting is not None and lhs_contracting == rhs_contracting:
                    out_mapping = list(lhs_map[:-1]) + [rhs_map[-1]]
                    out_mapping = [m if m != lhs_contracting else None for m in out_mapping]
                    node.sharding = SPMDShardingAnnotation(mesh, out_mapping)
                    if node.attributes.get("requires_scatter") or node.attributes.get("requires_reduce_scatter"):
                        node.attributes["inject_collective"] = "ReduceScatter"
                    else:
                        node.attributes["inject_collective"] = "AllReduce"
                    modified = True
                elif lhs_map[-2] is not None and rhs_contracting is None:
                    out_mapping = list(lhs_map[:-1]) + [rhs_map[-1]]
                    node.sharding = SPMDShardingAnnotation(mesh, out_mapping)
                    modified = True
                elif rhs_map[-1] is not None and lhs_contracting is None:
                    out_mapping = list(lhs_map[:-1]) + [rhs_map[-1]]
                    node.sharding = SPMDShardingAnnotation(mesh, out_mapping)
                    modified = True
                else:
                    node.sharding = lhs_sharding or rhs_sharding
                    modified = True

        # 3. Reductions: ReduceSum, ReduceMean, etc.
        elif (op_type in reduction_ops or op_type.startswith("Reduce") or op_type == "Sum") and len(node.inputs) >= 1:
            inp = graph.nodes.get(node.inputs[0])
            inp_sharding = getattr(inp, "sharding", None) if inp else None
            if inp_sharding is not None:
                inp_map = list(getattr(inp_sharding, "mesh_mapping", []))
                mesh = getattr(inp_sharding, "mesh", None)
                reduce_axis = node.attributes.get("axis", node.attributes.get("axes", 0))
                if isinstance(reduce_axis, (list, tuple)):
                    reduce_axes = [int(a) for a in reduce_axis]
                else:
                    reduce_axes = [int(reduce_axis)]

                is_reduced_dim_sharded = False
                out_map = []
                for dim_idx, m in enumerate(inp_map):
                    norm_idx = dim_idx if dim_idx >= 0 else dim_idx + len(inp_map)
                    if norm_idx in [a if a >= 0 else a + len(inp_map) for a in reduce_axes]:
                        if m is not None:
                            is_reduced_dim_sharded = True
                    else:
                        out_map.append(m)

                if not out_map:
                    out_map = [None]

                node.sharding = SPMDShardingAnnotation(mesh, out_map)
                if is_reduced_dim_sharded:
                    if node.attributes.get("requires_scatter") or node.attributes.get("requires_reduce_scatter"):
                        node.attributes["inject_collective"] = "ReduceScatter"
                    else:
                        node.attributes["inject_collective"] = "AllReduce"
                modified = True

        # 4. Spatial Convolutions
        elif op_type in conv_ops and len(node.inputs) >= 1:
            inp = graph.nodes.get(node.inputs[0])
            inp_sharding = getattr(inp, "sharding", None) if inp else None
            if inp_sharding is not None:
                node.sharding = inp_sharding
                modified = True

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
            coll_id = f"{node_id}_all_reduce" if coll_type == "AllReduce" else f"{node_id}_{coll_type.lower()}"
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
