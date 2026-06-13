"""Module docstring."""

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass


def test_shape_inference_pass_coverage() -> None:
    """Function docstring."""
    graph = IRGraph()

    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n2 = IRNode(
        id="n2",
        op_type="Reshape",
        inputs=["n1"],
        attributes={"newshape": (10,)},
        shape_metadata=(10,),
    )

    graph.nodes = {n.id: n for n in [n1, n2]}
    graph.inputs = ["n1"]
    graph.outputs = ["n2"]

    # We must mock get_op so it throws an exception that is NOT KeyError
    from unittest.mock import patch

    with (
        patch(
            "ml_switcheroo_compiler.transforms.passes.shape_inference.get_op",
            side_effect=ValueError("Test"),
        ),
        pytest.raises(CompilationError),
    ):
        shape_inference_pass(graph)

    # Also test successful paths
    graph2 = IRGraph()
    n3 = IRNode(
        id="n3",
        op_type="Constant",
        inputs=[],
        attributes={"value": [1.0, 2.0]},
        shape_metadata=None,
    )
    n4 = IRNode(id="n4", op_type="Output", inputs=["n3"], attributes={}, shape_metadata=None)
    n5 = IRNode(
        id="n5",
        op_type="Output",
        inputs=[],
        attributes={},
        shape_metadata=None,
    )  # No inputs
    n6 = IRNode(id="n6", op_type="Expand", inputs=["n3"], attributes={}, shape_metadata=(3, 2))
    n7 = IRNode(id="n7", op_type="BroadcastTo", inputs=["n3"], attributes={}, shape_metadata=(2, 2))
    n8 = IRNode(id="n8", op_type="Reshape", inputs=["n3"], attributes={}, shape_metadata=(2,))
    n9 = IRNode(id="n9", op_type="Unknown", inputs=["n3"], attributes={}, shape_metadata=None)

    graph2.nodes = {n.id: n for n in [n3, n4, n5, n6, n7, n8, n9]}

    assert shape_inference_pass(graph2)


def test_shape_inference_pass_coverage_2() -> None:
    """Function docstring."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n2 = IRNode(
        id="n2",
        op_type="Reshape",
        inputs=["n1"],
        attributes={"newshape": (10,)},
        shape_metadata=None,
    )
    graph.nodes = {n.id: n for n in [n1, n2]}
    from unittest.mock import MagicMock, patch

    mock_op = MagicMock()
    mock_op.infer_shape.return_value = (10,)
    with patch(
        "ml_switcheroo_compiler.transforms.passes.shape_inference.get_op",
        return_value=lambda: mock_op,
    ):
        shape_inference_pass(graph)
        assert graph.nodes["n2"].shape_metadata == (10,)
