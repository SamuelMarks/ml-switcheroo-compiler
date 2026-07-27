import sys
from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.config import config

mod = sys.modules["ml_switcheroo_compiler.ops.linalg.matmul"]


class DummyTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(shape)
        self.dtype = "float32"
        self.device = None

    @property
    def __class__(self):
        class TensorClass:
            __name__ = "Tensor"

        return TensorClass


def test_matmul_tracing():
    original_linalg = mod._emit_linalg_node
    original_shape = mod._emit_shape_node
    original_add = mod.add
    original_multiply = mod.multiply

    mock_linalg = MagicMock()
    mock_shape = MagicMock()
    mock_add = MagicMock()
    mock_multiply = MagicMock()

    mod._emit_linalg_node = mock_linalg
    mod._emit_shape_node = mock_shape
    mod.add = mock_add
    mod.multiply = mock_multiply

    config.eager_mode = False

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    mod.matmul(a, b)
    mod.dot(a, b)
    mod.vdot(a, b)
    mod.inner(a, b)
    mod.outer(a, b)
    mod.dot_general(a, b, (((1,), (0,)), ((), ())))

    mod.convolve(a, b)
    mod.matvec(a, b)
    mod.multi_dot([a, b])
    mod.vecdot(a, b)

    mod.addmm(a, a, b)
    mod.addmm(a, a, b, alpha=2.0, beta=2.0)

    mod.block_masked_mm(a, b, masks={"mask_out": a, "mask_lhs": a, "mask_rhs": a})
    mod.block_masked_mm(a, b)
    mod.gather_mm(a, b, lhs_indices=c, rhs_indices=c)
    mod.gather_mm(a, b, lhs_indices=None, rhs_indices=c)
    mod.gather_mm(a, b)
    mod.segmented_mm(a, b, c)

    mod._emit_linalg_node = original_linalg
    mod._emit_shape_node = original_shape
    mod.add = original_add
    mod.multiply = original_multiply


def test_matmul_eager():
    original_backend = mod.get_active_backend

    mock_get_backend = MagicMock()
    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 2))
    mock_backend.array.return_value = np.zeros((2, 2))
    mock_backend.array.return_value.shape = (2, 2)
    mock_get_backend.return_value = mock_backend

    mod.get_active_backend = mock_get_backend

    config.eager_mode = True

    a = DummyTensor((2, 2))
    b = DummyTensor((2, 2))
    c = DummyTensor((2,))

    mod.matmul(a, b)
    mod.dot(a, b)
    mod.vdot(a, b)
    mod.inner(a, b)
    mod.outer(a, b)
    mod.dot_general(a, b, (((1,), (0,)), ((), ())))

    mod.convolve(a, b)
    mod.multi_dot([a, b])
    mod.vecdot(a, b)

    mod.block_masked_mm(a, b, masks={"mask_out": a, "mask_lhs": a, "mask_rhs": a})
    mod.block_masked_mm(a, b)
    mod.gather_mm(a, b, lhs_indices=c, rhs_indices=c)
    mod.gather_mm(a, b)
    mod.segmented_mm(a, b, c)

    config.eager_mode = False
    mod.get_active_backend = original_backend


def test_matmul_opdefs():
    # BlockMaskedMm
    op1 = mod.BlockMaskedMm()
    assert op1.infer_shape((2, 2), (2, 2)) == (2, 2)
    try:
        with patch("ml_switcheroo_compiler.ops.linalg.matmul.matmul_shape", side_effect=ValueError):
            op1.infer_shape((2, 2), (3, 3))
    except Exception:
        pass
    assert op1.infer_shape("not_tuple", "not_tuple") == ()

    # GatherMm
    op2 = mod.GatherMm()
    assert op2.infer_shape("not_tuple", "not_tuple") == ()
    try:
        with patch("ml_switcheroo_compiler.ops.linalg.matmul.matmul_shape", side_effect=ValueError):
            op2.infer_shape((2, 2), (3, 3))
    except Exception:
        pass
    assert op2.infer_shape((2, 2), (2, 2), lhs_indices=(3,)) == (3, 2, 2)
    assert op2.infer_shape((2, 2), (2, 2), rhs_indices=(3,)) == (3, 2, 2)
    assert op2.infer_shape((2, 2), (2, 2)) == (2, 2)

    # SegmentedMm
    op3 = mod.SegmentedMm()
    assert op3.infer_shape("not_tuple", "not_tuple") == ()
    assert op3.infer_shape((2, 2), (2, 2)) == (1, 2, 2)
    assert op3.infer_shape((2, 2), (2, 2), segments=(3,)) == (2, 2, 2)
