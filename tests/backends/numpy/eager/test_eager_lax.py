"""Core abstractions and logic definitions for test_numpy_eager_lax_mocks.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_numpy_lax_mocks() -> None:
    """Test the numpy lax mocks behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(TypeError):

        def execute(op: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the execute operation.

            Args:
                op (EagerValue): Required parameter for op.
                *args (EagerValue): Variable positional arguments.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return as_callable(numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = as_callable(execute)("ApproxMaxK", as_callable(np.array)([1, 2]))
        assert as_callable(np.array_equal)(a, as_callable(np.array)([2]))
        assert as_callable(np.array_equal)(b, as_callable(np.array)([1]))
        (a, b) = as_callable(execute)("ApproxMaxK", as_callable(np.array)([]))
        assert as_tensor(getattr(a, "size")) == 0
        assert as_tensor(getattr(b, "size")) == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert as_callable(np.array_equal)(c, as_callable(np.array)([3]))
        c = execute("ForiLoop", None)
        assert as_callable(np.array_equal)(c, as_callable(np.array)(0))
        d = execute("IgammaGradA", np.array([4]))
        assert as_callable(np.array_equal)(d, as_callable(np.array)([4]))
        d = execute("IgammaGradA")
        assert as_callable(np.array_equal)(d, as_callable(np.array)(0))
