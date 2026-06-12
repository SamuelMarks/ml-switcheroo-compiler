"""Unit tests for verifying dynamic numpy-backed shape resolution and broadcasting in the.

interpreter.
"""


def test_shape_inference_broadcasting() -> None:
    """Verifies that the interpreter correctly infers and broadcasts shapes using a numpy-.

    backed execution

    This test constructs a logical graph with an 'Expand' operation, executes it
    with a 1D input array, and asserts that the output shape and values are
    correctly broadcasted to the target 2D shape

    Returns:
    None
    """
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo.interpreter import evaluate_graph

    g = LogicalGraph(outputs=["exp"])
    g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    g.nodes["exp"] = LogicalNode(
        id="exp",
        op_type="Expand",
        inputs=["x"],
        shape_metadata=(2, 3),
    )

    res = evaluate_graph(g, {"x": np.array([1, 2, 3])})
    assert res["exp"].shape == (2, 3)
    np.testing.assert_array_equal(res["exp"], np.array([[1, 2, 3], [1, 2, 3]]))
