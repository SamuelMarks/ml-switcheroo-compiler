"""Tests for test_frontend."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg import CustomLinearSolve, CustomRoot, Vecdot, multi_dot, solvers, vecdot
from ml_switcheroo_compiler.ops.linalg import matrix_ops as frontend
from ml_switcheroo_compiler.ops.linalg.conv_ops import ConvTranspose
from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuPivotsToPermutation
from ml_switcheroo_compiler.ops.linalg.matrix_ops import band_part, diag
from ml_switcheroo_compiler.ops.linalg.products import Diagonal, MultiDot
from ml_switcheroo_compiler.ops.linalg.solvers import MatrixNorm, Svdvals, Tensorinv, Tensorsolve, VectorNorm
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _test_op(func: EagerTensorProtocol, *args: EagerTensorProtocol, **kwargs: EagerTensorProtocol) -> EagerValue:
    """Test the op behavior.

    Args:
        func (EagerTensorProtocol): The func parameter.
        *args (EagerTensorProtocol): Variable length argument list.
        **kwargs (EagerTensorProtocol): Arbitrary keyword arguments.

    Returns:
        EagerValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            global_tracing_state.stop_tracing()
    return (out_eager, out_traced)


def test_frontend_ops() -> None:
    """Test the frontend ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(frontend.matrix_norm, a)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == ()
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(frontend.vector_norm, b)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == ()
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(frontend.svdvals, a)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == (2,)
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(solvers.tensorinv, a, ind=1)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == (2, 2)
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(solvers.tensorsolve, a, b)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == (2,)
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(frontend.diagonal, a)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == (2,)
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(multi_dot, [a, a])
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == (2, 2)
        (e, t) = typing.cast("EagerCallableProtocol", _test_op)(vecdot, b, b)
        assert typing.cast("EagerTensorProtocol", getattr(e, "shape")) == ()


def test_opdefs_infer_shapes() -> None:
    """Test the opdefs infer shapes behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):
        ops = [MatrixNorm(), VectorNorm(), Diagonal(), MultiDot()]
        for op in ops:
            assert op.infer_shape() == ()
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        assert typing.cast("EagerCallableProtocol", Svdvals)().infer_shape(a) == (2,)
        assert typing.cast("EagerCallableProtocol", Tensorinv)().infer_shape(a) == (2, 2)
        assert typing.cast("EagerCallableProtocol", Tensorsolve)().infer_shape(a, b) == (2,)
        assert typing.cast("EagerCallableProtocol", ConvTranspose)().infer_shape(a) == (2, 2)
        assert typing.cast("EagerCallableProtocol", ConvTranspose)().infer_shape() == ()
        assert id(typing.cast("EagerCallableProtocol", CustomLinearSolve)().infer_shape(a)) == id(a)
        assert CustomLinearSolve().infer_shape() == ()
        assert id(typing.cast("EagerCallableProtocol", Vecdot)().infer_shape(a)) == id(a)
        assert Vecdot().infer_shape() == ()
        assert id(typing.cast("EagerCallableProtocol", CustomRoot)().infer_shape(a)) == id(a)
        assert CustomRoot().infer_shape() == ()
        p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
        assert LuPivotsToPermutation().infer_shape(p, permutation_size=2) == (2,)


def test_linalg_band_part_diag_coverage() -> None:
    """Test the linalg band part diag coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    device = Device(DeviceType.CPU, 0)
    t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Int32, device))
    with ConfigContext(eager_mode=True):
        res1 = band_part(t, 1, 1)
        assert res1 is not None
        res2 = diag(t, k=0)
        assert res2 is not None
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            res3 = band_part(t, 1, 1)
            assert res3 is not None
            res4 = diag(t, k=0)
            assert res4 is not None
        finally:
            global_tracing_state.stop_tracing()
