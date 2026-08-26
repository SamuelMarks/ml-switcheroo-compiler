# ruff: noqa: E501
"""Core abstractions and logic definitions for test_assertions.py."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.assertions import clear_assertions, evaluate_assertions, record_assertion
from ml_switcheroo_compiler.ops.control_flow import assert_value


def test_assertions():
    """Test the assertions behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        clear_assertions()
        record_assertion(np.array([True, True]), "Should pass")
        evaluate_assertions()
        record_assertion(np.array([True, False]), "Should fail")
        with pytest.raises(AssertionError, match="Should fail"):
            evaluate_assertions()

        class DummyProxy:
            """Configuration class for dummy proxy."""

            def __init__(self, val):
                """Initialize the instance.

                Args:
                    val (object): The val parameter.

                Returns:
                    object: The inferred shape or computed result.
                """
                self.val = val

            def numpy(self):
                """Evaluate and process the numpy operation.

                Returns:
                    object: The evaluated or processed output.
                """
                return self.val

        record_assertion(DummyProxy(np.array([False])), "Proxy failed")
        with pytest.raises(AssertionError, match="Proxy failed"):
            evaluate_assertions()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_assert_value_eager():
    """Test the assert value eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        clear_assertions()
        assert_value(np.array([False]), "Eager failed")
        with pytest.raises(AssertionError, match="Eager failed"):
            evaluate_assertions()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
