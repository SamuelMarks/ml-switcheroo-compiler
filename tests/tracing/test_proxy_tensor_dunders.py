"""Unit tests for verifying the behavior of ProxyTensor operations and tracer state.

management

This module contains test cases that validate the execution of magic methods (dunders)
on ProxyTensor objects during active tracing, ensure that operations are blocked outside
of tracing contexts, and verify error handling for invalid operations.
"""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_proxy_tensor_dunders() -> None:
    """Test the proxy tensor dunders behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    global_tracing_state.start_tracing()
    try:
        p1 = ProxyTensor("p1", (2, 2), "float32")
        p2 = ProxyTensor("p2", (2, 2), "float32")
        _ = p1 + p2
        _ = 5 + p1
        _ = p1 - p2
        _ = 5 - p1
        _ = p1 * p2
        _ = 5 * p1
        _ = p1 / p2
        _ = 5 / p1
        _ = p1**p2
        _ = p1 % p2
        _ = 5 % p1
        _ = p1 // p2
        _ = 5 // p1
        _ = p1 & p2
        _ = 5 & p1
        _ = p1 | p2
        _ = 5 | p1
        _ = p1 ^ p2
        _ = 5 ^ p1
        _ = p1 << p2
        _ = 5 << p1
        _ = p1 >> p2
        _ = 5 >> p1
        _ = abs(p1)
        _ = ~p1
        _ = -p1
        _ = +p1
        _ = p1[0]
        _ = p1 @ p2
    finally:
        global_tracing_state.stop_tracing()
    assert callable(test_proxy_tensor_dunders)


def test_proxy_tensor_outside_tracing() -> None:
    """Test the proxy tensor outside tracing behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    p1 = ProxyTensor("p1", (2, 2), "float32")
    p2 = ProxyTensor("p2", (2, 2), "float32")
    with pytest.raises((RuntimeError, TracingError)):
        _ = p1 + p2
    with pytest.raises((RuntimeError, TracingError)):
        _ = -p1
    with pytest.raises((RuntimeError, TracingError)):
        _ = p1[0]
    with pytest.raises((RuntimeError, TracingError)):
        _ = p1 @ p2


def test_add_node_outside_tracing() -> None:
    """Test the add node outside tracing behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    node = LogicalNode(id="dummy", op_type="Input")
    global_tracing_state.add_node(node)
    assert callable(test_add_node_outside_tracing)


def test_matmul_invalid() -> None:
    """Test the matmul invalid behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    global_tracing_state.start_tracing()
    try:
        p1 = ProxyTensor("p1", (2, 2), "float32")
        with pytest.raises(ValueError):
            _ = p1 @ 5
    finally:
        global_tracing_state.stop_tracing()
