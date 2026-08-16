from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.transforms.passes.shape_inference import _determine_node_shape


def test_shape_inference_keyerror_explicit(monkeypatch):
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args, **kwargs):
        raise KeyError("foo")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(50, 60))
    assert _determine_node_shape(node, {}) == (50, 60)
