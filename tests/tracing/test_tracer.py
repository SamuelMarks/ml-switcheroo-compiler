# ruff: noqa: E501
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

"Core abstractions and logic definitions for test_tracer_coverage.py."


def test_proxy_tensor_binary_op() -> None:
    """Test the correctness and edge cases of the proxy tensor binary op functionality."""
    global_tracing_state.start_tracing()
    p1 = ProxyTensor(id="p1", shape=(2,), dtype="float32")
    p2 = ProxyTensor(id="p2", shape=(2,), dtype="float32")
    out = p1._binary_op(p2, "Add")
    assert out.shape == (2,)
    out_scalar = p1._binary_op(1.0, "Add")
    assert out_scalar.shape == (2,)
    out_unary = p1._unary_op("Neg")
    assert out_unary.shape == (2,)
    global_tracing_state.stop_tracing()
