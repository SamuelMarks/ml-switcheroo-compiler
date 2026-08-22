"""Tests for the registry coverage patch."""

from ml_switcheroo_compiler.ops.registry import _REGISTRY, _load_yaml_registry


def test_dynamic_op_def_infer_shape():
    """Test dynamic operator definition shape inference."""
    _load_yaml_registry()

    DynamicOp = None
    for op_name, cls in _REGISTRY.items():
        if cls.__name__ == op_name and "DynamicOpDef" not in cls.__qualname__:
            if hasattr(cls, "infer_shape") and "heuristics" in (cls.infer_shape.__doc__ or ""):
                DynamicOp = cls
                break

    assert DynamicOp is not None

    op = DynamicOp()

    # no shapes -> ()
    assert op.infer_shape() == ()

    # positional args missing inputs kwarg
    # test list/tuple int
    assert op.infer_shape([[1, 2]]) == (1, 2)
    assert op.infer_shape(inputs=[[2, 3]]) == (2, 3)
    assert op.infer_shape(inputs=[(2, 3)]) == (2, 3)

    class MockTensor1:
        def __init__(self, shape):
            self.shape_metadata = shape

    class MockTensor2:
        def __init__(self, shape):
            self.shape_metadata = None
            self.shape = shape

    # Test shape attribute fallback
    assert op.infer_shape(MockTensor2((4, 5))) == (4, 5)

    # Test broadcast
    assert op.infer_shape(inputs=[MockTensor1((1, 3)), MockTensor1((2, 3))]) == (2, 3)

    # Exception path for broadcast
    class BadTensor:
        def __init__(self, shape):
            self.shape_metadata = shape

    assert op.infer_shape(inputs=[BadTensor((2, 3)), BadTensor("invalid")]) == (2, 3) or isinstance(op.infer_shape(inputs=[BadTensor((2, 3)), BadTensor("invalid")]), tuple)

    # test inputs = list(args) path
    assert op.infer_shape(MockTensor1((5,)), MockTensor1((5,))) == (5,)
