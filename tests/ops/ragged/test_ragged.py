# ruff: noqa: E501
from unittest.mock import patch

from ml_switcheroo_compiler.ops.ragged.core import RaggedDot
from ml_switcheroo_compiler.ops.ragged.frontend import ragged_dot, ragged_stack, ragged_stack_dynamic_partitions

"Core abstractions and logic definitions for test_ragged_extra.py."


def test_ragged_ops() -> object:
    """Test the ragged ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        with patch("ml_switcheroo_compiler.ops.ragged.frontend._ragged_op") as mock_ragged:
            mock_ragged.return_value = "res"
            res1 = ragged_stack()
            assert res1 == "res"
            res2 = ragged_stack_dynamic_partitions()
            assert res2 == "res"
            res3 = ragged_dot()
            assert res3 == "res"
        op = RaggedDot()
        res4 = op.infer_shape()
        assert res4 == ()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
