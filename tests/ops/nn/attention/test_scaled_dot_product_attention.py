"""Behavioral tests for `scaled_dot_product_attention` in eager and IR tracing modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.shape import ShapeProtocol
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention, scaled_dot_product_attention
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import MockTestStub, ensure_tensor, require_callable


class _ShapedDummy(MockTestStub):
    """Operand exposing `.shape`."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize shaped operand.

        Args:
            shape (tuple[int, ...]): Desired shape tuple.
        """
        self.shape = shape


class _UnshapedDummy(MockTestStub):
    """Operand without `.shape`."""

    val: int = 42


def test_scaled_dot_product_attention_infer_shape() -> None:
    """Verify `ScaledDotProductAttention.infer_shape` and `dispatch_infer_shape` handle shaped and unshaped inputs."""
    op = ScaledDotProductAttention()
    assert op.infer_shape(_ShapedDummy((4, 8, 16)), None, None, None) == (4, 8, 16)
    assert op.infer_shape(_UnshapedDummy(), None, None, None) == ()

    reg_op = get_op("ScaledDotProductAttention")()
    assert reg_op.dispatch_infer_shape(q=_ShapedDummy((4, 4)), k=None, v=None, scale_factor=None) == (4, 4)


def test_scaled_dot_product_attention_eager_modes() -> None:
    """Verify eager `scaled_dot_product_attention` with `Tensor`s, raw arrays/scalars, and unshaped backend outputs."""
    q1 = Tensor(np.array([[[1.0]]], dtype=np.float32), TensorConfig((1, 1, 1), "float32", "cpu"))
    k1 = Tensor(np.array([[[1.0]]], dtype=np.float32), TensorConfig((1, 1, 1), "float32", "cpu"))
    v1 = Tensor(np.array([[[2.0]]], dtype=np.float32), TensorConfig((1, 1, 1), "float32", "cpu"))
    s1 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))

    with ConfigContext(backend="numpy", eager_mode=True):
        out_real = scaled_dot_product_attention(q1, k1, v1, s1)
        assert isinstance(out_real, Tensor)
        np.testing.assert_allclose(np.asarray(out_real.data), np.array([[[2.0]]], dtype=np.float32))

        q = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))
        k = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))
        v = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))
        sf = Tensor(np.array(0.25, dtype=np.float32), TensorConfig((), "float32", "cpu"))

        raw_arr = np.ones((2, 4, 8), dtype=np.float32)
        out_mixed = ensure_tensor(require_callable(scaled_dot_product_attention)(q, raw_arr, raw_arr, 0.25))
        assert isinstance(out_mixed, Tensor)
        assert out_mixed.shape == (2, 4, 8)

        backend = MagicMock()
        backend.execute_op.return_value = _UnshapedDummy()
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=backend):
            out_noshape = scaled_dot_product_attention(q, k, v, sf)
            assert isinstance(out_noshape, Tensor)
            assert out_noshape.shape == (2, 4, 8)


def test_scaled_dot_product_attention_tracing_ir_emission() -> None:
    """Verify `scaled_dot_product_attention` emits a `ScaledDotProductAttention` `LogicalNode` during tracing."""
    q = Tensor(ProxyTensor("q_in", (2, 4, 8), "float32"), TensorConfig((2, 4, 8), "float32", "cpu"))
    k = Tensor(ProxyTensor("k_in", (2, 4, 8), "float32"), TensorConfig((2, 4, 8), "float32", "cpu"))
    v = Tensor(ProxyTensor("v_in", (2, 4, 8), "float32"), TensorConfig((2, 4, 8), "float32", "cpu"))
    sf = Tensor(ProxyTensor("sf_in", (), "float32"), TensorConfig((), "float32", "cpu"))

    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("sdpa_trace")
        try:
            out = scaled_dot_product_attention(q, k, v, sf)
        finally:
            tape.stop_tracing()

    assert isinstance(out, ShapeProtocol)
    assert tuple(out.shape) == (2, 4, 8)
    sdpa_nodes = [n for n in traced_graph.nodes.values() if n.op_type == "ScaledDotProductAttention"]
    assert len(sdpa_nodes) == 1
    assert sdpa_nodes[0].inputs == ["q_in", "k_in", "v_in", "sf_in"]
