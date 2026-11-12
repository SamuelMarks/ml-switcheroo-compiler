"""Tests for LinearOperator structured matrix OpDef shape inference and eager evaluation."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.linear_operator import (
    LinearOperatorDiag,
    LinearOperatorFullMatrix,
    LinearOperatorIdentity,
)


def test_linear_operator_shape_inference_and_dispatch() -> None:
    """Verify LinearOperator OpDef shape inference for positional and keyword operands."""
    with ConfigContext(eager_mode=True):
        dev = Device("cpu")
        mat = Tensor(np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32), TensorConfig((2, 2), DType.Float32, dev))
        diag_vec = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, dev))

        assert LinearOperatorFullMatrix().infer_shape(mat) == (2, 2)
        assert LinearOperatorDiag().infer_shape(operand=diag_vec) == (2,)
        assert LinearOperatorIdentity().infer_shape(operator=mat) == (2, 2)
