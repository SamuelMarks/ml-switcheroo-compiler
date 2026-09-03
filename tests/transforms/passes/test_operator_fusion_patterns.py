def test_operator_fusion_extra():
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import MemoryAwareCostModel

    # Missing 295
    model = MemoryAwareCostModel(None)
    assert model.is_fusion_valid({}) is True

    # Missing 307-313
    model = MemoryAwareCostModel({"max_fusion_memory_bytes": 100, "memory_sizes": {"float32": 4}})
    node = IRNode("n", "Add", shape_metadata=(10, 10))
    # 10 * 10 * 4 = 400 > 100 -> False
    assert model.is_fusion_valid({"n": node}) is False

    node2 = IRNode("n2", "Add", shape_metadata=(2, 2))
    # 2 * 2 * 4 = 16 <= 100 -> True
    assert model.is_fusion_valid({"n2": node2}) is True
