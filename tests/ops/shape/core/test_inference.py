from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape_inference import _SHAPE_INFERENCE_REGISTRY, infer_shape, register_shape_inference


def test_register_shape_inference():
    @register_shape_inference("TestShapeOp")
    def my_infer_shape(*args, **kwargs):
        return (2, 3)

    assert _SHAPE_INFERENCE_REGISTRY["TestShapeOp"] == my_infer_shape
    assert infer_shape("TestShapeOp") == (2, 3)


def test_infer_shape_fallback():
    @register_op("TestShapeOp2")
    class TestShapeOp2(OpDef):
        def infer_shape(self, *args, **kwargs):
            return (4, 5)

    assert infer_shape("TestShapeOp2") == (4, 5)


def test_infer_shape_empty(mocker):
    class NoShapeOp:
        pass

    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=NoShapeOp)
    assert infer_shape("TestShapeOp3") == ()
