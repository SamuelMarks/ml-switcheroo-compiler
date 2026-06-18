from ml_switcheroo_compiler.ir.core import IRNode


def test_ir_node_sharding():
    # Verify that IRNode can carry sharding attribute.
    node = IRNode("node_1", "Add")
    node.sharding = "shard_info"

    # ensure it is serialized in kwargs or just held
    assert hasattr(node, "sharding")
    assert node.sharding == "shard_info"
