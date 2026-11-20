"""Unit tests for random tensor sequence shuffle transformations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.transformations import shuffle
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import ensure_tensor


def test_shuffle_ir_node_emission() -> None:
    """Verify `shuffle` emits `RandomShuffle` IR nodes preserving shape, dtype, inputs, and axis."""
    x_traced = Tensor(
        ProxyTensor(id="x_input", shape=(4, 5), dtype="float64"),
        TensorConfig((4, 5), DType.Float64, "cpu"),
    )

    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        tape.start_tracing("trace_shuffle_tensor")
        try:
            shuffled = shuffle("prng_key", x_traced, axis=1)
        finally:
            graph_tensor = tape.stop_tracing()

        assert graph_tensor is not None
        assert len(graph_tensor.nodes) == 1
        node = next(iter(graph_tensor.nodes.values()))
        assert node.op_type == "RandomShuffle"
        assert node.inputs == ["prng_key", "x_input"]
        assert node.shape_metadata == (4, 5)
        assert node.attributes == {"axis": 1}
        assert shuffled.shape == (4, 5)
        assert shuffled.dtype == DType.Float64

        tape.start_tracing("trace_shuffle_scalar_fallback")
        try:
            fallback_out = ensure_tensor(shuffle("key", "x"))
        finally:
            graph_fallback = tape.stop_tracing()

        assert graph_fallback is not None
        assert len(graph_fallback.nodes) == 1
        fb_node = next(iter(graph_fallback.nodes.values()))
        assert fb_node.op_type == "RandomShuffle"
        assert fb_node.inputs == ["key", "x"]
        assert fb_node.shape_metadata == ()
        assert fb_node.attributes == {"axis": 0}
        assert fallback_out.shape == ()
        assert fallback_out.dtype == DType.Float32
