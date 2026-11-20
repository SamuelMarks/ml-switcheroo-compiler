"""Behavioral tests for linear algebra IR node emission and shape/dtype extraction utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape


def test_emit_linalg_node_outside_tracing_raises() -> None:
    """Verify `_emit_linalg_node` raises `RuntimeError` when invoked outside an active tracing session."""
    with ConfigContext(eager_mode=False):
        inputs = [Tensor(np.ones((2, 2), dtype=np.float32), TensorConfig((2, 2), "float32", "cpu"))]
        with pytest.raises(RuntimeError, match="Cannot emit MatrixSolve node outside of a tracing context."):
            _emit_linalg_node("MatrixSolve", inputs, {}, [[2, 2]], [DType.Float32])


def test_emit_linalg_node_single_and_multi_output_tracing() -> None:
    """Verify `_emit_linalg_node` emits `LogicalNode`s and returns a `Tensor` or tuple of `Tensor`s."""
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("linalg_emit_trace")
        try:
            lhs = Tensor(ProxyTensor("mat_a", (2, 2), "float32"), TensorConfig((2, 2), "float32", "cpu"))
            rhs = Tensor(ProxyTensor("mat_b", (2, 2), "float32"), TensorConfig((2, 2), "float32", "cpu"))

            single_out = _emit_linalg_node(
                "MatrixSolve",
                [lhs, rhs],
                {"adjoint": False},
                out_shapes=[[2, 2]],
                out_dtypes=[DType.Float32],
            )
            multi_out = _emit_linalg_node(
                "QR",
                [lhs],
                {"full_matrices": False},
                out_shapes=[[2, 2], [2, 2]],
                out_dtypes=["float32", "float32"],
            )
        finally:
            tape.stop_tracing()

    assert isinstance(single_out, Tensor)
    assert single_out.shape == (2, 2)
    assert single_out.dtype == DType.Float32

    assert isinstance(multi_out, tuple)
    assert len(multi_out) == 2
    assert all(t.shape == (2, 2) for t in multi_out)

    op_types = {node.op_type for node in traced_graph.nodes.values()}
    assert {"MatrixSolve", "QR"} <= op_types
