# ruff: noqa: E501
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.dtype_inference import _get_promoted_dtype, _get_value_dtype, _handle_cast_dtype, _infer_constant_dtype, _infer_input_dtype, _infer_op_dtype, _infer_output_dtype, dtype_inference_pass

"Test Dtype Inference Pass."


class DummyVal:
    def __init__(self, dt: str) -> None:
        self.dtype = dt


def test_get_value_dtype() -> None:
    assert _get_value_dtype(DummyVal("float16")) == DType.Float16
    assert _get_value_dtype(True) == DType.Bool
    assert _get_value_dtype(1) == DType.Int32
    assert _get_value_dtype(1.0) == DType.Float32
    assert _get_value_dtype(None) is None


def test_infer_constant_dtype() -> None:
    node1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 1.0})
    dtypes: dict[str, str] = {}
    assert _infer_constant_dtype(node1, dtypes) is True
    assert dtypes["n1"] == DType.Float32.value
    node2 = IRNode(id="n2", op_type="Constant", inputs=[], attributes={"value": 1.0, "dtype": DType.Float32.value})
    assert _infer_constant_dtype(node2, dtypes) is False
    node3 = IRNode(id="n3", op_type="Constant", inputs=[], attributes={"dtype": DType.Int32.value})
    assert _infer_constant_dtype(node3, dtypes) is False
    assert dtypes["n3"] == DType.Int32.value


def test_infer_input_dtype() -> None:
    node1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={})
    dtypes: dict[str, str] = {}
    assert _infer_input_dtype(node1, dtypes) is False
    assert dtypes["n1"] == DType.Float32.value


def test_infer_output_dtype() -> None:
    node1 = IRNode(id="n1", op_type="Output", inputs=["in1"], attributes={})
    dtypes = {"in1": DType.Int32.value}
    assert _infer_output_dtype(node1, dtypes) is True
    assert dtypes["n1"] == DType.Int32.value
    node2 = IRNode(id="n2", op_type="Output", inputs=["in1"], attributes={"dtype": DType.Int32.value})
    assert _infer_output_dtype(node2, dtypes) is False


def test_get_promoted_dtype() -> None:
    assert _get_promoted_dtype(["float32"]) == "float32"
    assert _get_promoted_dtype(["int32", "float32"]) == "float32"
    assert _get_promoted_dtype(["int32", "float32", "complex64"]) == "complex64"
    assert _get_promoted_dtype(["unknown1", "unknown2"]) == "unknown1"


def test_handle_cast_dtype() -> None:
    node1 = IRNode(id="n1", op_type="Cast", inputs=[], attributes={"dtype": DType.Int32})
    assert _handle_cast_dtype(node1, []) == DType.Int32.value
    node2 = IRNode(id="n2", op_type="Cast", inputs=[], attributes={"dtype": "float16"})
    assert _handle_cast_dtype(node2, []) == "float16"
    node3 = IRNode(id="n3", op_type="Cast", inputs=[], attributes={})
    assert _handle_cast_dtype(node3, []) is None


def test_infer_op_dtype() -> None:
    dtypes = {"in1": DType.Int32.value, "in2": DType.Float32.value}
    node1 = IRNode(id="n1", op_type="Cast", inputs=["in1"], attributes={"dtype": DType.Float16})
    assert _infer_op_dtype(node1, dtypes) is True
    assert dtypes["n1"] == DType.Float16.value
    node2 = IRNode(id="n2", op_type="Equal", inputs=["in1", "in2"], attributes={})
    assert _infer_op_dtype(node2, dtypes) is True
    assert dtypes["n2"] == DType.Bool.value
    node3 = IRNode(id="n3", op_type="Add", inputs=["in1", "in2"], attributes={})
    assert _infer_op_dtype(node3, dtypes) is True
    assert dtypes["n3"] == DType.Float32.value
    node4 = IRNode(id="n4", op_type="Add", inputs=["in3"], attributes={})
    assert _infer_op_dtype(node4, dtypes) is True
    assert dtypes["n4"] == DType.Float32.value


def test_dtype_inference_pass() -> None:
    node1 = IRNode(id="in1", op_type="Constant", inputs=[], attributes={"value": 1})
    node2 = IRNode(id="in2", op_type="Constant", inputs=[], attributes={"value": 1.0})
    node3 = IRNode(id="add", op_type="Add", inputs=["in1", "in2"], attributes={})
    graph = IRGraph(name="test", nodes={"in1": node1, "in2": node2, "add": node3}, outputs=["add"])
    assert dtype_inference_pass(graph) is True
    assert graph.nodes["add"].attributes["dtype"] == DType.Float32.value


def test_dtype_inference_branches() -> None:
    node = IRNode(id="n1", op_type="Output", inputs=[], attributes={})
    dtypes: dict[str, str] = {}
    _infer_output_dtype(node, dtypes)
    assert dtypes["n1"] is None
    node2 = IRNode(id="n2", op_type="Add", inputs=["in1"], attributes={"dtype": DType.Float32.value})
    dtypes2 = {"in1": DType.Float32.value}
    res = _infer_op_dtype(node2, dtypes2)
    assert res is False
    node3 = IRNode(id="in1", op_type="Constant", inputs=[], attributes={"value": 1.0, "dtype": DType.Float32.value})
    graph = IRGraph(name="test", nodes={"in1": node3}, outputs=[])
    res2 = dtype_inference_pass(graph)
    assert res2 is False
