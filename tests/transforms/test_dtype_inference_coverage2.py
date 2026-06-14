"""Module docstring."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass


def test_dtype_inference_except_branch(monkeypatch: object) -> None:
    """Docstring."""
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Constant", inputs=[], attributes={"dtype": DType.Int32.value})
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"dtype": DType.Float32.value})
    n2 = IRNode(id="n2", op_type="Constant", inputs=[], attributes={"dtype": DType.Float64.value})
    n3 = IRNode(id="n3", op_type="Add", inputs=["n0", "n1", "n2"], attributes={})
    graph.nodes["n0"] = n0
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.nodes["n3"] = n3

    import ml_switcheroo_compiler.transforms.passes.dtype_inference as di

    def mock_promote(*args: object) -> object:
        """Docstring."""
        raise ValueError("mock error")

    monkeypatch.setattr(di, "promote_types", mock_promote)

    dtype_inference_pass(graph)
    assert graph.nodes["n3"].attributes["dtype"] == DType.Int32.value


def test_dtype_inference_3_inputs() -> None:
    """Docstring."""
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Constant", inputs=[], attributes={"dtype": DType.Int32.value})
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"dtype": DType.Float32.value})
    n2 = IRNode(id="n2", op_type="Constant", inputs=[], attributes={"dtype": DType.Float64.value})
    n3 = IRNode(id="n3", op_type="Add", inputs=["n0", "n1", "n2"], attributes={})
    graph.nodes["n0"] = n0
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.nodes["n3"] = n3
    dtype_inference_pass(graph)
    assert graph.nodes["n3"].attributes["dtype"] == DType.Float64.value
