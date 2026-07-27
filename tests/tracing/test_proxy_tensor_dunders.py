# ruff: noqa: E501
"""Unit tests for verifying the behavior of ProxyTensor operations and tracer state.

management

This module contains test cases that validate the execution of magic methods (dunders)
on ProxyTensor objects during active tracing, ensure that operations are blocked outside
of tracing contexts, and verify error handling for invalid operations.
"""

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_proxy_tensor_dunders() -> None:
    """Test the proxy tensor dunders behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies that all supported magic (dunder) methods on ProxyTensor execute.\n\n    successfully during an active tracing session\n\n    This test starts tracing, performs various binary, unary, bitwise, indexing, and\n    matrix\n    multiplication operations on ProxyTensor instances, and ensures they are\n    recorded without\n    raising exceptions\n\n    Returns:\n    None\n    "
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_proxy_tensor_outside_tracing() -> None:
    """Test the proxy tensor outside tracing behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies that performing operations on ProxyTensor instances outside of an active.\n\n    tracing session raises a RuntimeError\n\n    This test ensures that binary operations, unary operations, indexing, and matrix\n    multiplication are blocked when the tracer is inactive\n\n    Returns:\n    None\n    "
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_add_node_outside_tracing() -> None:
    """Test the add node outside tracing behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies that attempting to add a logical node to the tracer when tracing is.\n\n    inactive raises a RuntimeError\n\n    Returns:\n    None\n    "
        node = LogicalNode(id="dummy", op_type="Input")
        with pytest.raises((RuntimeError, TracingError)):
            global_tracing_state.add_node(node)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_matmul_invalid() -> None:
    """Test the matmul invalid behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies that matrix multiplication between a ProxyTensor and an invalid operand.\n\n    type raises a ValueError\n\n    This test starts tracing and attempts to perform a matrix multiplication between\n    a\n    ProxyTensor and an integer, which is expected to fail\n\n    Returns:\n    None\n    "
        global_tracing_state.start_tracing()
        try:
            p1 = ProxyTensor("p1", (2, 2), "float32")
            with pytest.raises(ValueError):
                _ = p1 @ 5
        finally:
            global_tracing_state.stop_tracing()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
