# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.spmd import _create_all_gather_node, _create_reduce_scatter_node, _inject_all_gather, _inject_reduce_scatter, _is_boundary_transition, _process_spmd_input, _process_spmd_node, inject_spmd_communication_pass


class DummySharding:
    def __init__(self, mapping: list) -> None:
        self.mesh_mapping = mapping


def test_is_boundary_transition() -> None:
    assert _is_boundary_transition(DummySharding([None]), DummySharding([None])) == (False, False)
    assert _is_boundary_transition(DummySharding(["x"]), DummySharding([None])) == (True, False)
    assert _is_boundary_transition(DummySharding([None]), DummySharding(["x"])) == (False, True)


def test_create_nodes() -> None:
    sharding = DummySharding(["x"])
    n1 = _create_all_gather_node("inp1", sharding)
    assert n1.id == "inp1_all_gather"
    assert n1.op_type == "all_gather"
    n2 = _create_reduce_scatter_node("inp2", sharding)
    assert n2.id == "inp2_reduce_scatter"
    assert n2.op_type == "reduce_scatter"


def test_inject_nodes() -> None:
    sharding = DummySharding(["x"])
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"])
    res1 = _inject_all_gather(node1, 0, "in1", sharding)
    assert node1.inputs[0] == "in1_all_gather"
    assert res1.id == "in1_all_gather"
    node2 = IRNode(id="n2", op_type="Add", inputs=["in1", "in2"])
    res2 = _inject_reduce_scatter(node2, 1, "in2", sharding)
    assert node2.inputs[1] == "in2_reduce_scatter"
    assert res2.id == "in2_reduce_scatter"


def test_process_spmd_input() -> None:
    sharding_unsharded = DummySharding([None])
    sharding_sharded = DummySharding(["x"])
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1"])
    graph = IRGraph(name="test", nodes={}, outputs=[])
    assert _process_spmd_input(node1, 0, "in1", graph, sharding_sharded) is None
    in_node1 = IRNode(id="in1", op_type="Input", inputs=[])
    graph.nodes["in1"] = in_node1
    assert _process_spmd_input(node1, 0, "in1", graph, sharding_sharded) is None
    in_node1.sharding = sharding_sharded
    res_ag = _process_spmd_input(node1, 0, "in1", graph, sharding_unsharded)
    assert res_ag is not None
    assert res_ag.op_type == "all_gather"
    in_node2 = IRNode(id="in2", op_type="Input", inputs=[], sharding=sharding_unsharded)
    node_grad = IRNode(id="ng", op_type="Grad", inputs=["in2"], sharding=sharding_sharded)
    graph.nodes["in2"] = in_node2
    res_rs = _process_spmd_input(node_grad, 0, "in2", graph, sharding_sharded)
    assert res_rs is not None
    assert res_rs.op_type == "reduce_scatter"
    node_other = IRNode(id="no", op_type="Add", inputs=["in2"], sharding=sharding_sharded)
    assert _process_spmd_input(node_other, 0, "in2", graph, sharding_sharded) is None


def test_process_spmd_node() -> None:
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[]), "in2": IRNode(id="in2", op_type="Input", inputs=[])}, outputs=[])
    (mod, inj) = _process_spmd_node(node1, graph)
    assert mod is False
    assert inj == []
    node1.sharding = DummySharding([None])
    graph.nodes["in1"].sharding = DummySharding(["x"])
    graph.nodes["in2"].sharding = DummySharding([None])
    (mod, inj) = _process_spmd_node(node1, graph)
    assert mod is True
    assert len(inj) == 1
    assert inj[0].op_type == "all_gather"


def test_inject_spmd_communication_pass() -> None:
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1"], sharding=DummySharding([None]))
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], sharding=DummySharding(["x"])), "n1": node1}, outputs=[])
    res = inject_spmd_communication_pass(graph)
    assert res is True
    assert "in1_all_gather" in graph.nodes
