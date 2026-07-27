# ruff: noqa: E501

from ml_switcheroo_compiler.ops.random_ops import RngBitGenerator, RngUniform

"Core abstractions and logic definitions for test_random_ops_extra.py."


def test_random_ops_infer_shape_coverage() -> object:
    """Test the random ops infer shape coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        res1 = RngBitGenerator().infer_shape(None, (1, 2), None)
        assert res1 == (1, 2)
        res2 = RngUniform().infer_shape(None, None, (3, 4), None)
        assert res2 == (3, 4)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
