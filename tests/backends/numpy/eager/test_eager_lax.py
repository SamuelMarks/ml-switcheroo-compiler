"""Core abstractions and logic definitions for test_numpy_eager_lax_mocks.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import require_callable, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_numpy_lax_mocks() -> None:
    """Test the numpy lax mocks behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):

        def execute(op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Evaluate and process the execute operation.

            Args:
                op (BackendArg): Required parameter for op.
                *args (BackendArg): Variable positional arguments.
                **kwargs (BackendArg): Arbitrary keyword arguments.

            Returns:
                BackendArg: The evaluated or processed output.
            """
            return require_callable(numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = unpack_pair(execute("ApproxMaxK", np.array([1, 2])))
        assert np.array_equal(ensure_ndarray(a), np.array([2]))
        assert np.array_equal(ensure_ndarray(b), np.array([1]))
        (a, b) = unpack_pair(execute("ApproxMaxK", np.array([])))
        assert ensure_ndarray(a).size == 0
        assert ensure_ndarray(b).size == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert np.array_equal(ensure_ndarray(c), np.array([3]))
        c = execute("ForiLoop", None)
        assert np.array_equal(ensure_ndarray(c), np.array(0))
        d = execute("IgammaGradA", np.array([4]))
        assert np.array_equal(ensure_ndarray(d), np.array([4]))
        d = execute("IgammaGradA")
        assert np.array_equal(ensure_ndarray(d), np.array(0))
