from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv_ops import (
    ConvGeneralDilated,
    ConvGeneralDilatedLocal,
    ConvGeneralDilatedPatches,
    Convolve,
    ConvTranspose,
    ConvTransposeShapeTuple,
    ConvWithGeneralPadding,
)


class DummyTensor:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "float32"
        self.config = None


def test_conv_ops_infer_shape():
    # ConvGeneralDilated
    op = ConvGeneralDilated()
    lhs = DummyTensor((1, 1))
    rhs = DummyTensor((1, 1))
    config = ConvConfig(window_strides=[], padding=[])

    op.infer_shape(lhs, rhs, config)
    op.infer_shape(lhs=lhs, rhs=rhs, config=config)
    op.infer_shape(lhs, rhs)
    op.infer_shape(lhs, rhs=rhs, config=config)

    # Missing shapes
    class NoShape:
        pass

    op.infer_shape(NoShape(), NoShape())

    # Convolve
    op2 = Convolve()
    op2.infer_shape(lhs, rhs)

    # ConvGeneralDilatedLocal
    op3 = ConvGeneralDilatedLocal()
    op3.infer_shape(lhs)
    op3.infer_shape()

    # ConvGeneralDilatedPatches
    op4 = ConvGeneralDilatedPatches()
    op4.infer_shape(lhs)
    op4.infer_shape()

    # ConvWithGeneralPadding
    op5 = ConvWithGeneralPadding()
    op5.infer_shape(lhs)
    op5.infer_shape()

    # ConvTransposeShapeTuple
    op6 = ConvTransposeShapeTuple()
    op6.infer_shape()

    # ConvTranspose
    op7 = ConvTranspose()
    lhs = DummyTensor((1, 4, 4, 3))
    rhs = DummyTensor((3, 3, 3, 2))
    assert op7.infer_shape(lhs, rhs, strides=2, padding="VALID") == (1, 9, 9, 2)
    assert op7.infer_shape(lhs, rhs, strides=2, padding="SAME") == (1, 8, 8, 2)
    assert op7.infer_shape(lhs, rhs, strides=(2, 2), padding="VALID") == (1, 9, 9, 2)
