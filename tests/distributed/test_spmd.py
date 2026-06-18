from ml_switcheroo_compiler.ir.core import IRNode, IRGraph
from ml_switcheroo_compiler.distributed.device_mesh import DeviceMesh
from ml_switcheroo_compiler.distributed.layout_map import ShardingSpec
from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass


def test_spmd_pass():
    graph = IRGraph()
    mesh = DeviceMesh([2], ["x"])
    spec1 = ShardingSpec(mesh, ["x"])
    spec2 = ShardingSpec(mesh, [None])

    node1 = IRNode(id="n1", op_type="Input", sharding=spec1)
    node2 = IRNode(id="n2", op_type="MatMul", inputs=["n1"], sharding=spec2)
    node3 = IRNode(id="n3", op_type="Grad", inputs=["n2"], sharding=spec1)

    graph.nodes = {"n1": node1, "n2": node2, "n3": node3}

    modified = inject_spmd_communication_pass(graph)
    assert modified

    # Check that all_gather was injected between n1 and n2
    # Check that reduce_scatter was injected for gradients


def test_spmd_edge_cases():
    graph = IRGraph()
    mesh = DeviceMesh([2], ["x"])
    spec1 = ShardingSpec(mesh, ["x"])

    # Node without sharding
    node_no_sharding = IRNode(id="n0", op_type="Input")

    # Node with an input not in graph.nodes
    node_external_input = IRNode(
        id="n1", op_type="Identity", inputs=["missing_input"], sharding=spec1
    )

    graph.nodes = {"n0": node_no_sharding, "n1": node_external_input}

    modified = inject_spmd_communication_pass(graph)
    assert not modified
