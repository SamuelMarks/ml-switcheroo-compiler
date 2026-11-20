"""Behavioral tests for `space_to_batch` and `space_to_batch_nd` in eager and IR tracing modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.shape.space_batch as space_batch_mod
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.space_batch import SpaceToBatch, SpaceToBatchND, space_to_batch, space_to_batch_nd
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import extract_test_shape


def test_space_to_batch_op_infer_shape() -> None:
    """Verify `SpaceToBatch` and `SpaceToBatchND` default `infer_shape` returns empty tuple when unshaped."""
    assert SpaceToBatch().infer_shape(None, 2, [[0, 0], [0, 0]]) == ()
    assert SpaceToBatchND().infer_shape(None, [2, 2], [[0, 0], [0, 0]]) == ()


def test_space_to_batch_eager_execution() -> None:
    """Verify `space_to_batch` and `space_to_batch_nd` execute eagerly on shaped `Tensor` and raw list inputs."""
    t_in = Tensor(np.zeros((1, 4, 4, 1), dtype=np.float32), TensorConfig((1, 4, 4, 1), DType.Float32, Device("cpu")))
    raw_input = [[1, 2], [3, 4]]

    mock_backend = MagicMock()
    with (
        ConfigContext(eager_mode=True),
        patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend),
    ):
        mock_backend.execute_op.return_value = np.zeros((4, 2, 2, 1), dtype=np.float32)
        res1 = space_to_batch(t_in, 2, [[0, 0], [0, 0]])
        assert extract_test_shape(res1) == (4, 2, 2, 1)

        res_nd1 = space_to_batch_nd(t_in, [2, 2], [[0, 0], [0, 0]])
        assert extract_test_shape(res_nd1) == (4, 2, 2, 1)

        mock_backend.execute_op.return_value = 42
        res2 = space_batch_mod.space_to_batch(raw_input, 2, [[0, 0], [0, 0]])
        assert isinstance(res2, Tensor)
        assert res2.shape == ()

        res_nd2 = space_batch_mod.space_to_batch_nd(raw_input, [2, 2], [[0, 0], [0, 0]])
        assert isinstance(res_nd2, Tensor)
        assert res_nd2.shape == ()


def test_space_to_batch_symbolic_tracing_ir_emission() -> None:
    """Verify `space_to_batch` and `space_to_batch_nd` emit `SpaceToBatch` and `SpaceToBatchND` IR nodes."""
    t_sym = Tensor(
        ProxyTensor("sb_in", (1, 4, 4, 1), "float32"),
        TensorConfig((1, 4, 4, 1), DType.Float32, Device("cpu")),
    )
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("space_batch_trace")
        try:
            out_sb = space_to_batch(t_sym, 2, [[0, 0], [0, 0]])
            out_sb_nd = space_to_batch_nd(t_sym, [2, 2], [[0, 0], [0, 0]])
        finally:
            tape.stop_tracing()

    assert isinstance(out_sb, Tensor)
    assert isinstance(out_sb_nd, Tensor)
    op_types = {node.op_type for node in traced_graph.nodes.values()}
    assert {"SpaceToBatch", "SpaceToBatchND"} <= op_types
