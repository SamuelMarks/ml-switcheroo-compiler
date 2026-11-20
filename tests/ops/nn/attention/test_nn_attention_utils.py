"""Behavioral tests for RoPE, sinusoidal positional encoding, and ALiBi mask utilities."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, alibi_mask, rope, sinusoidal_positional_encoding
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import MockTestStub


class _ShapedDummy(MockTestStub):
    """Operand providing `.shape`."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize shaped dummy operand.

        Args:
            shape (tuple[int, ...]): Desired shape tuple.
        """
        self.shape = shape


class _UnshapedDummy(MockTestStub):
    """Operand lacking `.shape`."""

    val: int = 123


def test_attention_utils_infer_shape_and_eager_execution() -> None:
    """Verify `RopeOp.infer_shape` and eager NumPy evaluation of RoPE, positional encoding, and ALiBi mask."""
    rope_op = RopeOp()
    assert rope_op.infer_shape(_ShapedDummy((2, 4, 8))) == (2, 4, 8)
    assert rope_op.infer_shape(_UnshapedDummy()) == ()

    t_in = Tensor(
        np.ones((1, 2, 4), dtype=np.float32),
        TensorConfig(shape=(1, 2, 4), dtype=DType.Float32, device=Device("cpu")),
    )
    assert ScaledDotProductAttention().infer_shape(t_in, t_in, t_in, None) == (1, 2, 4)

    with ConfigContext(backend="numpy", eager_mode=True):
        rope_out = rope(t_in, axis=4)
        assert isinstance(rope_out, Tensor)
        assert rope_out.shape == (1, 2, 4)

        pe = sinusoidal_positional_encoding(seq_len=4, axis=8)
        assert isinstance(pe, Tensor)
        assert pe.shape == (4, 8)

        mask = alibi_mask(seq_len=4, num_heads=2)
        assert mask == 0.0


def test_rope_symbolic_tracing_ir_emission() -> None:
    """Verify `rope` emits a canonical `Rope` `LogicalNode` when traced with `TracerTape`."""
    t_sym = Tensor(
        ProxyTensor("rope_in", (2, 4, 8), "float32"),
        TensorConfig((2, 4, 8), DType.Float32, Device("cpu")),
    )
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("rope_trace")
        try:
            out = rope(t_sym, axis=8)
        finally:
            tape.stop_tracing()

    assert isinstance(out, Tensor)
    assert out.shape == (2, 4, 8)
    rope_nodes = [n for n in traced_graph.nodes.values() if n.op_type == "Rope"]
    assert len(rope_nodes) == 1
    assert rope_nodes[0].inputs == ["rope_in"]
    assert rope_nodes[0].attributes["axis"] == 8
