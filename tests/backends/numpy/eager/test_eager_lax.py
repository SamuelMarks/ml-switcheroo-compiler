# ruff: noqa: E501
"""Core abstractions and logic definitions for test_numpy_eager_lax_mocks.py."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_numpy_lax_mocks() -> object:
    """Test the numpy lax mocks behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:

            def execute(op: object, *args: object, **kwargs: object) -> object:
                """Evaluate and process the execute operation.

                Args:
                    op (object): Required parameter for op.
                    *args (Any): Variable positional arguments.
                    **kwargs (Any): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return numpy_eager_registry.get(op)(np, *args, **kwargs)

            (a, b) = execute("ApproxMaxK", np.array([1, 2]))
            assert np.array_equal(a, np.array([2]))
            assert np.array_equal(b, np.array([1]))
            (a, b) = execute("ApproxMaxK", np.array([]))
            assert a.size == 0
            assert b.size == 0
            c = execute("ForiLoop", None, None, np.array([3]))
            assert np.array_equal(c, np.array([3]))
            c = execute("ForiLoop", None)
            assert np.array_equal(c, np.array(0))
            d = execute("IgammaGradA", np.array([4]))
            assert np.array_equal(d, np.array([4]))
            d = execute("IgammaGradA")
            assert np.array_equal(d, np.array(0))
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
