"""Module test_rematerialization_extras.py."""


def test_rematerialization_max_dist():
    """test_rematerialization_max_dist."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.transforms.passes.rematerialization import _find_target_nodes

    node = LogicalNode(id="n1", op_type="A", inputs=[], shape_metadata=(1024, 1024), attributes={"dtype": "float32"})
    consumers = {"n1": ["n2"]}
    node_indices = {"n1": 0, "n2": 5}  # diff is 5, <= 10

    rules = {"target_ops": ["A"], "thresholds": {"min_memory_bytes": 0, "max_compute_to_memory_ratio": 100.0}}

    res = _find_target_nodes([node], consumers, node_indices, rules)
    assert not res
