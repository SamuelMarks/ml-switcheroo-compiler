"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_numpy_lax_mocks() -> object:
    """Function docstring."""

    def execute(op: object, *args: object, **kwargs: object) -> object:
        """Function docstring."""
        return numpy_eager_registry.get(op)(np, *args, **kwargs)

    # Test tuples
    a, b = execute("ApproxMaxK", np.array([1, 2]))
    assert np.array_equal(a, np.array([2]))
    assert np.array_equal(b, np.array([1]))

    a, b = execute("ApproxMaxK", np.array([]))
    assert a.size == 0
    assert b.size == 0

    # Test loops
    c = execute("ForiLoop", None, None, np.array([3]))
    assert np.array_equal(c, np.array([3]))

    c = execute("ForiLoop", None)
    assert np.array_equal(c, np.array(0))

    # Test fallback
    d = execute("IgammaGradA", np.array([4]))
    assert np.array_equal(d, np.array([4]))

    d = execute("IgammaGradA")
    assert np.array_equal(d, np.array(0))
