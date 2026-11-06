"""Core abstractions and logic definitions for test_numpy_eager_lax_mocks.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from tests.testing_utils import ensure_callable, ensure_eager_tensor

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
            return ensure_callable(numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = ensure_callable(execute)("ApproxMaxK", ensure_callable(np.array)([1, 2]))
        assert ensure_callable(np.array_equal)(a, ensure_callable(np.array)([2]))
        assert ensure_callable(np.array_equal)(b, ensure_callable(np.array)([1]))
        (a, b) = ensure_callable(execute)("ApproxMaxK", ensure_callable(np.array)([]))
        assert ensure_eager_tensor(ensure_eager_tensor(a).size) == 0
        assert ensure_eager_tensor(ensure_eager_tensor(b).size) == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert ensure_callable(np.array_equal)(c, ensure_callable(np.array)([3]))
        c = execute("ForiLoop", None)
        assert ensure_callable(np.array_equal)(c, ensure_callable(np.array)(0))
        d = execute("IgammaGradA", np.array([4]))
        assert ensure_callable(np.array_equal)(d, ensure_callable(np.array)([4]))
        d = execute("IgammaGradA")
        assert ensure_callable(np.array_equal)(d, ensure_callable(np.array)(0))
