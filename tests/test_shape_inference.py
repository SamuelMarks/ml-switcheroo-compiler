"""Test stubs for dynamic numpy-backed shape resolution."""


def test_shape_inference_broadcasting():
    """Test verifying dynamic numpy-backed shape resolution for broadcasting."""
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode
    from ml_switcheroo.interpreter import evaluate_graph

    g = LogicalGraph(outputs=["exp"])
    g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    g.nodes["exp"] = LogicalNode(
        id="exp", op_type="Expand", inputs=["x"], shape_metadata=(2, 3)
    )

    res = evaluate_graph(g, {"x": np.array([1, 2, 3])})
    assert res["exp"].shape == (2, 3)
    np.testing.assert_array_equal(res["exp"], np.array([[1, 2, 3], [1, 2, 3]]))
