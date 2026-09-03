def test_operator_fusion_memory_aware():
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import MemoryAwareCostModel

    config = {"max_fusion_memory_bytes": 1024, "memory_sizes": {"float32": 4}}
    cost = MemoryAwareCostModel(config)

    node = IRNode("mul1", "Multiply", ["a", "b"], {}, shape_metadata=[1024])
    node.attributes["dtype"] = "float32"
    assert cost.is_fusion_valid({"mul1": node}) is False

    node2 = IRNode("mul2", "Multiply", ["a", "b"], {}, shape_metadata=[10])
    node2.attributes["dtype"] = "float32"
    assert cost.is_fusion_valid({"mul2": node2}) is True

    node3 = IRNode("mul3", "Multiply", ["a", "b"], {}, shape_metadata=["symbolic"])
    assert cost.is_fusion_valid({"mul3": node3}) is True

    cost2 = MemoryAwareCostModel(None)
    assert cost2.is_fusion_valid({"mul1": node}) is True
