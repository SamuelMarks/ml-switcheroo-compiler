from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.spmd import (
    _create_all_gather_node,
    _create_all_reduce_node,
    _create_all_to_all_node,
    _create_reduce_scatter_node,
    _inject_all_gather,
    _inject_all_reduce,
    _inject_all_to_all,
    _inject_reduce_scatter,
    _is_boundary_transition,
    _process_spmd_input,
    _process_spmd_node,
    inject_spmd_communication_pass,
)


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
    assert n1.op_type == "AllGather"
    assert n1.attributes.get("dispatch_early") is True

    n2 = _create_reduce_scatter_node("inp2", sharding)
    assert n2.id == "inp2_reduce_scatter"
    assert n2.op_type == "ReduceScatter"
    assert n2.attributes.get("dispatch_early") is True

    n3 = _create_all_reduce_node("inp3", sharding)
    assert n3.id == "inp3_all_reduce"
    assert n3.op_type == "AllReduce"
    assert n3.attributes.get("dispatch_early") is True

    n4 = _create_all_to_all_node("inp4", sharding)
    assert n4.id == "inp4_all_to_all"
    assert n4.op_type == "AllToAll"
    assert n4.attributes.get("dispatch_early") is True


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

    node3 = IRNode(id="n3", op_type="Add", inputs=["in3", "in2"])
    res3 = _inject_all_reduce(node3, 0, "in3", sharding)
    assert node3.inputs[0] == "in3_all_reduce"
    assert res3.id == "in3_all_reduce"

    node4 = IRNode(id="n4", op_type="Add", inputs=["in4", "in2"])
    res4 = _inject_all_to_all(node4, 0, "in4", sharding)
    assert node4.inputs[0] == "in4_all_to_all"
    assert res4.id == "in4_all_to_all"


def test_process_spmd_input() -> None:
    sharding_unsharded = DummySharding([None])
    sharding_sharded_x = DummySharding(["x"])
    sharding_sharded_y = DummySharding(["y"])

    graph = IRGraph()
    in_node1 = IRNode(id="in1", op_type="Input", inputs=[], sharding=sharding_sharded_x)
    graph.nodes["in1"] = in_node1

    # 1. inp_sharded, not node_sharded, not grad/reduction => all_gather
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1"], sharding=sharding_unsharded)
    res_ag = _process_spmd_input(node1, 0, "in1", graph, sharding_unsharded)
    assert res_ag is not None and res_ag.op_type == "AllGather"

    # 2. inp_sharded, not node_sharded, reduction => all_reduce
    node2 = IRNode(id="n2", op_type="ReduceSum", inputs=["in1"], sharding=sharding_unsharded)
    res_ar = _process_spmd_input(node2, 0, "in1", graph, sharding_unsharded)
    assert res_ar is not None and res_ar.op_type == "AllReduce"

    # 3. not inp_sharded, node_sharded, grad => reduce_scatter
    in_node_unsharded = IRNode(id="in_un", op_type="Input", inputs=[], sharding=sharding_unsharded)
    graph.nodes["in_un"] = in_node_unsharded
    node_grad = IRNode(id="grad_node", op_type="Grad", inputs=["in_un"], sharding=sharding_sharded_x)
    res_rs = _process_spmd_input(node_grad, 0, "in_un", graph, sharding_sharded_x)
    assert res_rs is not None and res_rs.op_type == "ReduceScatter"

    # 4. inp_sharded, node_sharded, different axes => all_to_all
    node3 = IRNode(id="n3", op_type="Add", inputs=["in1"], sharding=sharding_sharded_y)
    res_a2a = _process_spmd_input(node3, 0, "in1", graph, sharding_sharded_y)
    assert res_a2a is not None and res_a2a.op_type == "AllToAll"

    # Missing input node
    node4 = IRNode(id="n4", op_type="Add", inputs=["missing"], sharding=sharding_sharded_y)
    assert _process_spmd_input(node4, 0, "missing", graph, sharding_sharded_y) is None

    # Missing sharding on input node
    in_node_no_shard = IRNode(id="in_ns", op_type="Input", inputs=[])
    graph.nodes["in_ns"] = in_node_no_shard
    assert _process_spmd_input(node3, 0, "in_ns", graph, sharding_sharded_y) is None


def test_process_spmd_node() -> None:
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[]), "in2": IRNode(id="in2", op_type="Input", inputs=[])})

    (mod, inj) = _process_spmd_node(node1, graph)
    assert not mod and not inj

    node1.sharding = DummySharding([None])
    graph.nodes["in1"].sharding = DummySharding(["x"])
    graph.nodes["in2"].sharding = DummySharding([None])
    (mod, inj) = _process_spmd_node(node1, graph)
    assert mod is True
    assert len(inj) == 1
    assert inj[0].op_type == "AllGather"


def test_inject_spmd_communication_pass() -> None:
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1"], sharding=DummySharding([None]))
    graph = IRGraph(nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], sharding=DummySharding(["x"])), "n1": node1})
    res = inject_spmd_communication_pass(graph)
    assert res is True
    assert "in1_all_gather" in graph.nodes


def test_get_sharding_axes_none() -> None:
    from ml_switcheroo_compiler.transforms.passes.spmd import _get_sharding_axes

    assert _get_sharding_axes(None) == []


def test_spmd_insert_communications_not_is_grad():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Add", inputs=["n2"], sharding=DummySharding([0]))
    n2 = IRNode(id="n2", op_type="Input", sharding=None)
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    res = inject_spmd_communication_pass(g)
    assert res is False  # Not is_grad, so it doesn't insert reduce_scatter


def test_spmd_insert_communications_axes_len_mismatch():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Add", inputs=["n2"], sharding=DummySharding([0]))
    n2 = IRNode(id="n2", op_type="Input", sharding=DummySharding([0, 1]))
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    res = inject_spmd_communication_pass(g)
    assert res is False  # Length of axes mismatch, no all_to_all


def test_spmd_insert_communications_not_inp_sharded_node_sharded_not_grad():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Add", inputs=["n2"], sharding=DummySharding([0]))
    n2 = IRNode(id="n2", op_type="Input", sharding=DummySharding([None]))
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    # Wait, n1 has node_sharded True, n2 has inp_sharded False.
    # But n1 is NOT grad.
    # Actually wait. _is_boundary_transition(inp, node) gives False, True.
    # _process_spmd_input does:
    # elif not inp_sharded and node_sharded:
    #     if is_grad: ...
    # So if is_grad is False, it returns None.
    res = inject_spmd_communication_pass(g)
    assert res is False


def test_spmd_empty_conditions(mocker):
    from ml_switcheroo_compiler.transforms.passes.spmd import _process_spmd_input

    # Create dummy rule with empty conditions
    mock_rules = {"reductions": [], "communication_matrix": [{"state": [True, True], "conditions": []}]}
    mocker.patch("ml_switcheroo_compiler.transforms.passes.spmd._get_spmd_rules", return_value=mock_rules)

    graph = IRGraph()
    in_node1 = IRNode(id="in1", op_type="Input", inputs=[], sharding=DummySharding(["x"]))
    graph.nodes["in1"] = in_node1

    node1 = IRNode(id="n1", op_type="Add", inputs=["in1"], sharding=DummySharding(["y"]))
    res = _process_spmd_input(node1, 0, "in1", graph, DummySharding(["y"]))
    assert res is None
