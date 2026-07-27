# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.lift_state import _get_nodes, _lift_block_ir, _lift_node, flatten_state_dict, lift_state_pass, unflatten_state_dict


def test_lift_state_branches() -> None:

    class Block:
        nodes = {"n1": IRNode(id="n1", op_type="Add", inputs=[], attributes={"some_attr": "no_nodes"})}

    _lift_block_ir(Block())


"Test Lift State Pass."


def test_flatten_unflatten_state_dict() -> None:
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_state_dict(nested)
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}
    assert unflatten_state_dict(flat) == nested


def test_get_nodes() -> None:

    class DummyBlock:
        nodes = {"n1": IRNode(id="n1", op_type="Input", inputs=[])}

    assert list(_get_nodes(DummyBlock()))[0].id == "n1"

    class DummyBlock2:
        nodes = [IRNode(id="n2", op_type="Input", inputs=[])]

    assert list(_get_nodes(DummyBlock2()))[0].id == "n2"

    class DummyBlock3:
        pass

    assert list(_get_nodes(DummyBlock3())) == []


def test_lift_node() -> None:

    class DummyBlock:
        outputs: list[str] = []

    block = DummyBlock()
    node1 = IRNode(id="n1", op_type="ReadVariable", inputs=[])
    assert _lift_node(node1, block) is True
    assert node1.op_type == "Input"
    node2 = IRNode(id="n2", op_type="AssignVariable", inputs=["in1"])
    assert _lift_node(node2, block) is True
    assert node2.op_type == "Output"
    assert node2.inputs == ["in1"]
    assert "n2" in block.outputs
    node3 = IRNode(id="n3", op_type="Assign", inputs=["var", "val"])
    assert _lift_node(node3, block) is True
    assert node3.op_type == "Output"
    assert node3.inputs == ["val"]
    assert "n3" in block.outputs
    node4 = IRNode(id="n4", op_type="Assign", inputs=["var2", "val2"])
    block.outputs = ["n4"]
    assert _lift_node(node4, block) is True
    assert node4.op_type == "Output"
    assert block.outputs == ["n4"]

    class BlockNoOutputs:
        pass

    node5 = IRNode(id="n5", op_type="Assign", inputs=["var3", "val3"])
    assert _lift_node(node5, BlockNoOutputs()) is True
    assert node5.op_type == "Output"
    node6 = IRNode(id="n6", op_type="Add", inputs=[])
    assert _lift_node(node6, block) is False


def test_lift_block_ir() -> None:

    class SubBlock:
        nodes = {"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}

    node1 = IRNode(id="n2", op_type="Cond", inputs=[], attributes={"body": SubBlock()})

    class Block:
        nodes = {"n2": node1}

    assert _lift_block_ir(Block()) is True
    assert SubBlock.nodes["n1"].op_type == "Input"


def test_lift_state_pass() -> None:
    graph = IRGraph(name="test", nodes={"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}, outputs=[])
    assert lift_state_pass(graph) is True
    assert graph.nodes["n1"].op_type == "Input"
