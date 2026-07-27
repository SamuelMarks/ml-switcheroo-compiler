from ml_switcheroo_compiler.ops.reductions.distributed import Pmean, Psum


class MockTensor:
    def __init__(self, shape):
        self.shape = shape


def test_psum_infer_shape():
    op = Psum()
    t = MockTensor((2, 3))

    # args
    assert op.infer_shape(t) == (2, 3)

    # kwargs
    assert op.infer_shape(x=t) == (2, 3)

    # empty
    assert op.infer_shape() == ()


def test_pmean_infer_shape():
    op = Pmean()
    t = MockTensor((2, 3))

    # args
    assert op.infer_shape(t) == (2, 3)

    # kwargs
    assert op.infer_shape(x=t) == (2, 3)

    # empty
    assert op.infer_shape() == ()
