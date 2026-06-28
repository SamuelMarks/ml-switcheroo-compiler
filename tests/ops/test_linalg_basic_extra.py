from unittest.mock import MagicMock

from ml_switcheroo_compiler.ops.linalg.basic import MatrixPower, Pinv


def test_linalg_infer_shape_coverage():
    mock_a = MagicMock()
    mock_a.shape = (2, 3)
    assert Pinv().infer_shape(mock_a) == (3, 2)

    mock_b = MagicMock()
    mock_b.shape = (5,)
    assert Pinv().infer_shape(mock_b) == (5,)

    assert MatrixPower().infer_shape(mock_a) == (2, 3)


def test_conv_general_dilated_emit_backends():
    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilated

    op = ConvGeneralDilated()
    assert op.emit_jax() == "Not implemented ConvGeneralDilated"
    assert op.emit_keras() == "Not implemented ConvGeneralDilated"
    assert op.emit_mlx() == "Not implemented ConvGeneralDilated"
    assert op.emit_pytorch() == "Not implemented ConvGeneralDilated"
    assert op.emit_tensorflow() == "Not implemented ConvGeneralDilated"


def test_fftnd_infer_shape_branches():
    from ml_switcheroo_compiler.ops.linalg.basic import Fftnd, Ifftnd, Rfftnd, Irfftnd
    import numpy as np
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.zeros((4, 4)), TensorConfig((4, 4), "float32", "cpu"))

    op_f = Fftnd()
    assert op_f.infer_shape(t) == (4, 4)

    op_i = Ifftnd()
    assert op_i.infer_shape(t) == (4, 4)

    op_r = Rfftnd()
    assert op_r.infer_shape(t) == (4, 3)

    op_ir = Irfftnd()
    assert op_ir.infer_shape(t) == (4, 6)


def test_fftnd_infer_shape_branches_empty_list():
    from ml_switcheroo_compiler.ops.linalg.basic import Fftnd, Ifftnd, Rfftnd, Irfftnd
    import numpy as np
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    t = Tensor(np.zeros((4, 4)), TensorConfig((4, 4), "float32", "cpu"))

    op_f = Fftnd()
    assert op_f.infer_shape(t, s=[], axes=[]) == (4, 4)

    op_i = Ifftnd()
    assert op_i.infer_shape(t, s=[], axes=[]) == (4, 4)

    op_r = Rfftnd()
    assert op_r.infer_shape(t, s=[], axes=[]) == (4, 4)

    op_ir = Irfftnd()
    assert op_ir.infer_shape(t, s=[], axes=[]) == (4, 4)
