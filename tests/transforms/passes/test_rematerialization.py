def test_rematerialization_pass():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.rematerialization import _estimate_compute, _estimate_memory, rematerialization_pass

    # test _estimate_memory
    n1 = IRNode(id="n1", op_type="Add")
    assert _estimate_memory(n1) == 4.0

    n2 = IRNode(id="n2", op_type="Add")
    n2.shape_metadata = 5
    assert _estimate_memory(n2) == 4.0

    n3 = IRNode(id="n3", op_type="Add")
    n3.shape_metadata = (10, 10)
    assert _estimate_memory(n3) == 400.0

    # test _estimate_compute
    assert _estimate_compute(n1, {}) == 1.0

    n4 = IRNode(id="n4", op_type="MatMul")
    n4.shape_metadata = (10, 10)
    assert _estimate_compute(n4, {"high_cost_ops": ["MatMul"]}) == 10000.0

    # test rematerialization_pass
    g = IRGraph()
    # Create a target op (e.g. Relu) that is big and has consumers far away
    n_input = IRNode(id="inp", op_type="Input")
    n_target = IRNode(id="targ", op_type="Relu", inputs=["inp"])
    n_target.shape_metadata = (1000, 1000)  # big enough mem

    nodes = {"inp": n_input, "targ": n_target}
    g.nodes.update(nodes)

    # add dummy nodes to increase max_dist > 10
    last_id = "targ"
    for i in range(12):
        new_id = f"dummy_{i}"
        g.nodes[new_id] = IRNode(id=new_id, op_type="Add", inputs=[last_id])
        last_id = new_id

    g.nodes["far_consumer"] = IRNode(id="far_consumer", op_type="Add", inputs=["targ"])

    from unittest.mock import patch

    mock_rules = {"target_ops": ["Relu"], "thresholds": {"min_memory_bytes": 1000, "max_compute_to_memory_ratio": 100.0}}
    with patch("ml_switcheroo_compiler.transforms.passes.rematerialization._load_rules", return_value=mock_rules):
        modified = rematerialization_pass(g)

    assert modified is True
    assert "targ_remat" in g.nodes
    assert g.nodes["far_consumer"].inputs == ["targ_remat"]
    assert g.nodes["targ"].attributes.get("rematerialize") is True


def test_rematerialization_pass_no_modify():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass

    g = IRGraph()
    # No targets
    n = IRNode(id="n1", op_type="Input")
    g.nodes["n1"] = n

    with patch("ml_switcheroo_compiler.transforms.passes.rematerialization._load_rules", return_value={"target_ops": []}):
        modified = rematerialization_pass(g)
        assert modified is False


def test_rematerialization_clone_exists():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass

    g = IRGraph()
    n_target = IRNode(id="targ", op_type="Relu", inputs=[])
    n_target.shape_metadata = (1000, 1000)

    # already cloned
    n_clone = IRNode(id="targ_remat", op_type="Relu", inputs=[])

    g.nodes["targ"] = n_target
    g.nodes["targ_remat"] = n_clone

    last_id = "targ"
    for i in range(12):
        new_id = f"dummy_{i}"
        g.nodes[new_id] = IRNode(id=new_id, op_type="Add", inputs=[last_id])
        last_id = new_id
    g.nodes["far_consumer"] = IRNode(id="far_consumer", op_type="Add", inputs=["targ"])

    mock_rules = {"target_ops": ["Relu"], "thresholds": {"min_memory_bytes": 1000, "max_compute_to_memory_ratio": 100.0}}
    with patch("ml_switcheroo_compiler.transforms.passes.rematerialization._load_rules", return_value=mock_rules):
        modified = rematerialization_pass(g)

    assert modified is True
    assert g.nodes["far_consumer"].inputs == ["targ_remat"]


def test_load_rules_real():
    from ml_switcheroo_compiler.transforms.passes.rematerialization import _load_rules

    rules = _load_rules()
    assert isinstance(rules, dict)
    assert "target_ops" in rules


def test_rematerialization_branch_coverage():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.rematerialization import _estimate_compute, rematerialization_pass

    # test _estimate_compute with int shape
    n_int = IRNode(id="n_int", op_type="Add")
    n_int.shape_metadata = 5
    assert _estimate_compute(n_int, {}) == 1.0

    g = IRGraph()
    n_input = IRNode(id="inp", op_type="Input")
    # target op that is big enough, but consumers are close
    n_target = IRNode(id="targ", op_type="Relu", inputs=["inp"])
    n_target.shape_metadata = (1000, 1000)

    n_close = IRNode(id="close", op_type="Add", inputs=["targ"])

    # target op that is big enough, but it has no consumers
    n_target2 = IRNode(id="targ2", op_type="Relu", inputs=["inp"])
    n_target2.shape_metadata = (1000, 1000)

    # Node with an input that is not in the graph
    n_missing = IRNode(id="n_missing", op_type="Add", inputs=["missing_input"])

    nodes = {"inp": n_input, "targ": n_target, "close": n_close, "targ2": n_target2, "n_missing": n_missing}
    g.nodes.update(nodes)

    mock_rules = {"target_ops": ["Relu"], "thresholds": {"min_memory_bytes": 1000, "max_compute_to_memory_ratio": 100.0}}
    with patch("ml_switcheroo_compiler.transforms.passes.rematerialization._load_rules", return_value=mock_rules):
        modified = rematerialization_pass(g)

    assert modified is False
