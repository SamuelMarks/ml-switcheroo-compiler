"""Module docstring."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.assertions import (
    clear_assertions,
    evaluate_assertions,
    record_assertion,
)
from ml_switcheroo_compiler.ops.control_flow import assert_value


def test_assertions() -> object:
    """Function docstring."""
    clear_assertions()
    record_assertion(np.array([True, True]), "Should pass")
    evaluate_assertions()  # Should not raise

    record_assertion(np.array([True, False]), "Should fail")
    with pytest.raises(AssertionError, match="Should fail"):
        evaluate_assertions()

    class DummyProxy:
        """Class docstring."""

        def __init__(self, val: object) -> object:
            """Function docstring."""
            self.val = val

        def numpy(self) -> object:
            """Function docstring."""
            return self.val

    record_assertion(DummyProxy(np.array([False])), "Proxy failed")
    with pytest.raises(AssertionError, match="Proxy failed"):
        evaluate_assertions()


def test_assert_value_eager() -> object:
    """Function docstring."""
    clear_assertions()
    assert_value(np.array([False]), "Eager failed")
    with pytest.raises(AssertionError, match="Eager failed"):
        evaluate_assertions()
