"""Unit tests for verifying the behavior of ProxyTensor operations and tracer state.

management

This module contains test cases that validate the execution of magic methods (dunders)
on ProxyTensor objects during active tracing, ensure that operations are blocked outside
of tracing contexts, and verify error handling for invalid operations.
"""

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_proxy_tensor_dunders() -> None:
    """Verifies that all supported magic (dunder) methods on ProxyTensor execute.

    successfully during an active tracing session

    This test starts tracing, performs various binary, unary, bitwise, indexing, and
    matrix
    multiplication operations on ProxyTensor instances, and ensures they are
    recorded without
    raising exceptions

    Returns:
    None
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


def test_proxy_tensor_outside_tracing() -> None:
    """Verifies that performing operations on ProxyTensor instances outside of an active.

    tracing session raises a RuntimeError

    This test ensures that binary operations, unary operations, indexing, and matrix
    multiplication are blocked when the tracer is inactive

    Returns:
    None
    """
    p1 = ProxyTensor("p1", (2, 2), "float32")
    p2 = ProxyTensor("p2", (2, 2), "float32")

    with pytest.raises(RuntimeError):
        _ = p1 + p2

    with pytest.raises(RuntimeError):
        _ = -p1

    with pytest.raises(RuntimeError):
        _ = p1[0]

    with pytest.raises(RuntimeError):
        _ = p1 @ p2


def test_add_node_outside_tracing() -> None:
    """Verifies that attempting to add a logical node to the tracer when tracing is.

    inactive raises a RuntimeError

    Returns:
    None
    """
    node = LogicalNode(id="dummy", op_type="Input")
    with pytest.raises(RuntimeError):
        global_tracing_state.add_node(node)


def test_matmul_invalid() -> None:
    """Verifies that matrix multiplication between a ProxyTensor and an invalid operand.

    type raises a ValueError

    This test starts tracing and attempts to perform a matrix multiplication between
    a
    ProxyTensor and an integer, which is expected to fail

    Returns:
    None
    """
    global_tracing_state.start_tracing()
    try:
        p1 = ProxyTensor("p1", (2, 2), "float32")
        with pytest.raises(ValueError):
            _ = p1 @ 5
    finally:
        global_tracing_state.stop_tracing()
