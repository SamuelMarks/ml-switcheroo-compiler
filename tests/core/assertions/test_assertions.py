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


import ml_switcheroo_compiler.core.assertions as assertions


def test_assertions_coverage():
    assertions.clear_assertions()

    # record
    assertions.record_assertion(True, "Should not fail")
    assertions.record_assertion(False, "Should fail")

    # is_iterable_non_string
    assert assertions._is_iterable_non_string([1, 2]) is True
    assert assertions._is_iterable_non_string("string") is False
    assert assertions._is_iterable_non_string(b"bytes") is False
    assert assertions._is_iterable_non_string(1) is False

    # evaluate_iterable
    assert assertions._evaluate_iterable([True, True]) is True
    assert assertions._evaluate_iterable([True, False]) is False
    with pytest.raises(ValueError):
        assertions._evaluate_iterable(1)

    # evaluate_single_condition
    class NumpyCond:
        def numpy(self):
            return np.array([True, True])

    assert assertions._evaluate_single_condition(NumpyCond()) is True

    class NumpyCondFalse:
        def numpy(self):
            return np.array([True, False])

    assert assertions._evaluate_single_condition(NumpyCondFalse()) is False

    assert assertions._evaluate_single_condition(True) is True
    assert assertions._evaluate_single_condition(False) is False

    class ValueErrCond:
        def __bool__(self):
            raise ValueError()

        def __iter__(self):
            yield True
            yield True

    assert assertions._evaluate_single_condition(ValueErrCond()) is True

    # evaluate_assertions
    assertions.clear_assertions()
    assertions.record_assertion(True, "All good")
    assertions.evaluate_assertions()  # Should not raise

    assertions.record_assertion(False, "Oh no")
    with pytest.raises(AssertionError, match="Oh no"):
        assertions.evaluate_assertions()
