"""Behavioral tests for `eye`, `identity`, and `diag` in eager and IR tracing modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.creation.api_matrix import _diag_eager, diag, eye, identity
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import MockTestStub, ensure_tensor, require_callable


def test_matrix_creation_eager_execution() -> None:
    """Verify `eye`, `identity`, `_diag_eager`, and `diag` compute exact matrices on the NumPy backend."""
    with ConfigContext(backend="numpy", eager_mode=True):
        e_sq = ensure_tensor(eye(3))
        np.testing.assert_allclose(np.asarray(e_sq.data), np.eye(3, dtype=np.float32))
        assert e_sq.shape == (3, 3)

        e_rect = ensure_tensor(eye(3, m=4, k=1, dtype=DType.Float64, device=Device("cpu")))
        np.testing.assert_allclose(np.asarray(e_rect.data), np.eye(3, 4, k=1, dtype=np.float64))
        assert e_rect.shape == (3, 4)
        assert e_rect.dtype == DType.Float64

        class _CustomDTypeName:
            """Custom dtype descriptor lacking `.value`."""

            shape: tuple[int, ...] = ()

            def __str__(self) -> str:
                """Return string representation `'float32'`.

                Returns:
                    str: Canonical dtype string.
                """
                return "float32"

        e_custom = ensure_tensor(require_callable(eye)(2, dtype=_CustomDTypeName()))
        assert e_custom.shape == (2, 2)

        ident = ensure_tensor(identity(4))
        np.testing.assert_allclose(np.asarray(ident.data), np.eye(4, dtype=np.float32))

        mat = Tensor(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            TensorConfig((2, 2), DType.Float32, Device("cpu")),
        )
        d_default = ensure_tensor(_diag_eager(mat, 0, Device("cpu"), None))
        np.testing.assert_allclose(np.asarray(d_default.data), np.array([1.0, 4.0], dtype=np.float32))

        d_eager = ensure_tensor(diag(mat, diagonal=1))
        np.testing.assert_allclose(np.asarray(d_eager.data), np.array([2.0], dtype=np.float32))

        scalar_backend = MagicMock()
        scalar_backend.execute_op.return_value = 123.0
        with patch("ml_switcheroo_compiler.ops.creation.api_matrix.get_active_backend", return_value=scalar_backend):
            d_scalar = ensure_tensor(_diag_eager(mat, 0, Device("cpu"), DType.Float32))
            assert d_scalar.shape == ()


def test_matrix_creation_symbolic_tracing_and_validation() -> None:
    """Verify `eye`, `identity`, and `diag` emit `EyeLike` and `Diag` IR nodes and validate rank/tracing."""
    t_1d = Tensor(np.array([1.0, 2.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    t_const_2d = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        TensorConfig((2, 2), DType.Float32, Device("cpu")),
    )

    with ConfigContext(eager_mode=False):
        with pytest.raises(ValueError, match="diag requires a 1D or 2D tensor."):
            diag(t_1d)

        with pytest.raises(RuntimeError, match="Cannot emit diag node outside of a tracing context."):
            diag(t_const_2d)

        tape = TracerTape()
        traced_graph = tape.start_tracing("matrix_creation_trace")
        try:
            sym_eye = ensure_tensor(eye(3, 4, k=1))
            sym_id = ensure_tensor(identity(2))

            t_proxy = Tensor(
                ProxyTensor("mat_in", (3, 3), "float32"),
                TensorConfig((3, 3), DType.Float32, Device("cpu")),
            )
            diag_proxy = ensure_tensor(diag(t_proxy, diagonal=1))
            diag_const = ensure_tensor(diag(t_const_2d, diagonal=0))

            diag_direct = ensure_tensor(require_callable(diag)(ProxyTensor("direct_mat_id", (4, 4), "float32"), 0))

            class _ShapeOnlyMat(MockTestStub):
                """Direct matrix input without `.data` or `.dtype` attributes."""

                shape: tuple[int, ...] = (4, 4)

            diag_nodtype = ensure_tensor(require_callable(diag)(_ShapeOnlyMat(), 0))
        finally:
            tape.stop_tracing()

    assert sym_eye.shape == (3, 4)
    assert sym_id.shape == (2, 2)
    assert diag_proxy.shape == (2,)
    assert diag_const.shape == (2,)
    assert diag_direct.shape == (4,)
    assert diag_nodtype.shape == (4,)

    eye_nodes = [n for n in traced_graph.nodes.values() if n.op_type == "EyeLike"]
    diag_nodes = [n for n in traced_graph.nodes.values() if n.op_type == "Diag"]
    assert len(eye_nodes) == 2
    assert len(diag_nodes) == 4
    assert any(n.inputs == ["mat_in"] and n.attributes == {"k": 1} for n in diag_nodes)
    assert any(n.inputs == ["const"] and n.attributes == {"k": 0} for n in diag_nodes)
    assert any(n.inputs == ["direct_mat_id"] for n in diag_nodes)
