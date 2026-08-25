from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
from ml_switcheroo_compiler.transforms.passes.lift_state import lift_state_pass


def test_ad_state_lifting_integration():
    """Verify that mutable state updates integrate securely into reverse-mode AD."""
    # Create a graph that reads state, computes loss, and assigns new state
    graph = IRGraph(name="test")
    # State read
    read_node = IRNode(id="state_val", op_type="ReadVariable", inputs=[])
    # Parameter
    param_node = IRNode(id="param", op_type="Input", inputs=[])
    # Computation
    loss_node = IRNode(id="loss", op_type="Multiply", inputs=["state_val", "param"])
    # State write
    write_node = IRNode(id="write_state", op_type="AssignVariable", inputs=["loss"])

    graph.nodes = {"state_val": read_node, "param": param_node, "loss": loss_node, "write_state": write_node}
    graph.inputs = ["param"]
    graph.outputs = ["loss"]

    # 1. Lift state
    lift_state_pass(graph)

    # Verify lift
    assert read_node.op_type == "Input"
    assert write_node.op_type == "Output"
    assert write_node.inputs == ["loss"]
    assert "write_state" in graph.outputs  # State assignments become additional outputs

    # 2. Integrate with AD
    # We want the gradient wrt 'param'.
    # 'loss' is the loss node.
    grad_graph = graph_grad(graph, wrt=["param"], output_id="loss")

    # Verify that AD ran successfully and produced gradient outputs
    assert len(grad_graph.outputs) == 1
    # AD should traverse from 'loss' back to 'param'
    # The gradient of Multiply(state_val, param) wrt param is state_val.

    # Find the output node in grad_graph
    out_node_id = grad_graph.outputs[0]
    # It should trace back to state_val
    assert any(n.id == "state_val" for n in grad_graph.nodes.values() if n.op_type == "Input")
