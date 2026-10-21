"""Tests for test_matmul."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
mod = sys.modules["ml_switcheroo_compiler.ops.linalg.matmul"]


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.data = as_callable(np.zeros)(shape)
        self.dtype = "float32"
        self.device = None

    if not typing.TYPE_CHECKING:

        @property
        def __class__(self) -> type:
            """Test function __class__."""

            class TensorClass:
                """Test class TensorClass."""

                __name__ = "Tensor"

            return TensorClass


def test_matmul_tracing() -> None:
    """Test function test_matmul_tracing."""
    original_linalg = mod._emit_linalg_node
    original_shape = mod._emit_shape_node
    original_add = mod.add
    original_multiply = mod.multiply
    mock_linalg = MagicMock()
    mock_shape = MagicMock()
    mock_add = MagicMock()
    mock_multiply = MagicMock()
    setattr(mod, "_emit_linalg_node", mock_linalg)
    setattr(mod, "_emit_shape_node", mock_shape)
    setattr(mod, "add", mock_add)
    setattr(mod, "multiply", mock_multiply)
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
    setattr(mod, "_emit_linalg_node", original_linalg)
    setattr(mod, "_emit_shape_node", original_shape)
    setattr(mod, "add", original_add)
    setattr(mod, "multiply", original_multiply)
    assert c.shape == (2,)


def test_matmul_eager() -> None:
    """Test function test_matmul_eager."""
    original_backend = mod.get_active_backend
    mock_get_backend = MagicMock()
    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.zeros((2, 2))
    mock_backend.array.return_value = np.zeros((2, 2))
    mock_backend.array.return_value.shape = (2, 2)
    mock_get_backend.return_value = mock_backend
    setattr(mod, "get_active_backend", mock_get_backend)
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
    setattr(mod, "get_active_backend", original_backend)
    assert c.shape == (2,)


def test_matmul_opdefs() -> None:
    """Test function test_matmul_opdefs."""
    op1 = mod.BlockMaskedMm()
    assert op1.infer_shape((2, 2), (2, 2)) == (2, 2)
    with pytest.raises(AttributeError):
        with patch("ml_switcheroo_compiler.ops.linalg.matmul.matmul_shape", side_effect=ValueError):
            op1.infer_shape((2, 2), (3, 3))
    assert op1.infer_shape("not_tuple", "not_tuple") == ()
    op2 = mod.GatherMm()
    assert op2.infer_shape("not_tuple", "not_tuple") == ()
    with pytest.raises(AttributeError):
        with patch("ml_switcheroo_compiler.ops.linalg.matmul.matmul_shape", side_effect=ValueError):
            op2.infer_shape((2, 2), (3, 3))
    assert op2.infer_shape((2, 2), (2, 2), lhs_indices=(3,)) == (3, 2, 2)
    assert op2.infer_shape((2, 2), (2, 2), rhs_indices=(3,)) == (3, 2, 2)
    assert op2.infer_shape((2, 2), (2, 2)) == (2, 2)
    op3 = mod.SegmentedMm()
    assert op3.infer_shape("not_tuple", "not_tuple") == ()
    assert op3.infer_shape((2, 2), (2, 2)) == (1, 2, 2)
    assert op3.infer_shape((2, 2), (2, 2), segments=(3,)) == (2, 2, 2)


def test_dot_general_missing_shape() -> None:
    """Test function test_dot_general_missing_shape."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    old_eager = config.eager_mode
    config.eager_mode = False
    try:

        class FakeData:
            """Test class FakeData."""

            shape = ()

        Tensor(data=FakeData(), config=TensorConfig((), DType.Float32, None))
        Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
                with patch("ml_switcheroo_compiler.ops.linalg.utils._emit_linalg_node", return_value="dummy_node"):
                    pass
    finally:
        config.eager_mode = old_eager
    assert config.eager_mode == old_eager


def test_dot_general_missing_shape_real() -> None:
    """Test function test_dot_general_missing_shape_real."""
    import sys

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    matmul_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.matmul"]
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = False
    try:

        class FakeData:
            """Test class FakeData."""

            shape = ()

        t1 = Tensor(data=FakeData(), config=TensorConfig((), DType.Float32, None))
        t2 = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
                original_emit = matmul_mod._emit_linalg_node
                setattr(matmul_mod, "_emit_linalg_node", lambda *args, **kwargs: "dummy_node")
                try:
                    assert matmul_mod.dot_general(t1, t2, (((), ()), ((), ()))) == "dummy_node"
                finally:
                    setattr(matmul_mod, "_emit_linalg_node", original_emit)
    finally:
        config.eager_mode = old_eager


def test_matmul_missing(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_matmul_missing.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.linalg.matmul import BlockMaskedMm, GatherMm, dot_general

    op1 = BlockMaskedMm()
    assert as_callable(op1.infer_shape)((2, 3), (4, 5)) is None
    op2 = GatherMm()
    assert as_callable(op2.infer_shape)((2, 3), (4, 5), lhs_indices=None, rhs_indices=None) is None
    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            t1 = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
            t2 = Tensor(np.array([3.0, 4.0]), TensorConfig((2,), "float32", "cpu"))
            res = dot_general(t1, t2, dimension_numbers=(((0,), (0,)), ((), ())))
            assert isinstance(res, Tensor)


def test_solvers_missing() -> None:
    """Test function test_solvers_missing."""
    from ml_switcheroo_compiler.ops.linalg.solvers import Pinv

    op = Pinv()
    assert as_callable(op.infer_shape)(None) == ()

    class DummyShape:
        """Test class DummyShape."""

        shape = (3,)

    assert as_callable(op.infer_shape)(as_callable(DummyShape)()) == (3,)


def test_einsum_missing() -> None:
    """Test function test_einsum_missing."""
    import pytest

    from ml_switcheroo_compiler.ops.linalg.einsum import ParsedEquationPart

    p = ParsedEquationPart(chars="ij", shape=(3,))
    with pytest.raises(ValueError):
        p._check_dimension_mismatch({"i": 2}, "i", 3)


def test_einsum_missing_branch() -> None:
    """Test function test_einsum_missing_branch."""
    from ml_switcheroo_compiler.ops.linalg.einsum import ParsedEquationPart

    p = ParsedEquationPart(chars="ij", shape=(3, 1))
    p.shape = (3, 1)
    setattr(p, "chars", ["i", "i"])
    axis_map = {"i": 3}
    p.process_axis_map(axis_map)
    assert axis_map["i"] == 3
