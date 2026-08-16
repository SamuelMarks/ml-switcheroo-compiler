import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.transforms.passes.shape_inference import _determine_node_shape


def test_shape_inference_keyerror():
    # Unknown op -> KeyError -> returns shape_metadata
    node = IRNode(id="n1", op_type="UnknownOp2", inputs=[], attributes={}, shape_metadata=(10, 20))
    assert _determine_node_shape(node, {}) == (10, 20)


def test_shape_inference_valueerror_not_found(monkeypatch):
    # If something raises ValueError with "Operation ... not found", it returns shape_metadata
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args, **kwargs):
        raise ValueError("Operation 'Foo' not found")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(30, 40))
    assert _determine_node_shape(node, {}) == (30, 40)


def test_shape_inference_valueerror_other(monkeypatch):
    # If something raises ValueError without "Operation not found", raises CompilationError
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args, **kwargs):
        raise ValueError("Some other error")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(30, 40))
    with pytest.raises(CompilationError):
        _determine_node_shape(node, {})
