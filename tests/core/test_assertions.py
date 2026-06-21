import pytest
import numpy as np
from ml_switcheroo_compiler.core.assertions import (
    record_assertion,
    evaluate_assertions,
    clear_assertions,
)
from ml_switcheroo_compiler.ops.control_flow import assert_value


def test_assertions():
    clear_assertions()
    record_assertion(np.array([True, True]), "Should pass")
    evaluate_assertions()  # Should not raise

    record_assertion(np.array([True, False]), "Should fail")
    with pytest.raises(AssertionError, match="Should fail"):
        evaluate_assertions()

    class DummyProxy:
        def __init__(self, val):
            self.val = val

        def numpy(self):
            return self.val

    record_assertion(DummyProxy(np.array([False])), "Proxy failed")
    with pytest.raises(AssertionError, match="Proxy failed"):
        evaluate_assertions()


def test_assert_value_eager():
    clear_assertions()
    assert_value(np.array([False]), "Eager failed")
    with pytest.raises(AssertionError, match="Eager failed"):
        evaluate_assertions()
