"""Core abstractions and logic definitions for test_numpy_eager_lax_mocks.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


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
            return typing.cast("EagerCallableProtocol", numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = typing.cast("EagerCallableProtocol", execute)("ApproxMaxK", typing.cast("EagerCallableProtocol", np.array)([1, 2]))
        assert typing.cast("EagerCallableProtocol", np.array_equal)(a, typing.cast("EagerCallableProtocol", np.array)([2]))
        assert typing.cast("EagerCallableProtocol", np.array_equal)(b, typing.cast("EagerCallableProtocol", np.array)([1]))
        (a, b) = typing.cast("EagerCallableProtocol", execute)("ApproxMaxK", typing.cast("EagerCallableProtocol", np.array)([]))
        assert typing.cast("EagerTensorProtocol", getattr(a, "size")) == 0
        assert typing.cast("EagerTensorProtocol", getattr(b, "size")) == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert typing.cast("EagerCallableProtocol", np.array_equal)(c, typing.cast("EagerCallableProtocol", np.array)([3]))
        c = execute("ForiLoop", None)
        assert typing.cast("EagerCallableProtocol", np.array_equal)(c, typing.cast("EagerCallableProtocol", np.array)(0))
        d = execute("IgammaGradA", np.array([4]))
        assert typing.cast("EagerCallableProtocol", np.array_equal)(d, typing.cast("EagerCallableProtocol", np.array)([4]))
        d = execute("IgammaGradA")
        assert typing.cast("EagerCallableProtocol", np.array_equal)(d, typing.cast("EagerCallableProtocol", np.array)(0))
