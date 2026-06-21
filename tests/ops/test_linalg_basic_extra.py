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
