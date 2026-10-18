"""Core abstractions and logic definitions for test_numpy_eager_lax_mocks.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
            return cast("Callable[..., EagerTensorProtocol]", numpy_eager_registry.get)(op)(np, *args, **kwargs)

        (a, b) = cast("Callable[..., EagerTensorProtocol]", execute)("ApproxMaxK", cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(a, cast("Callable[..., EagerTensorProtocol]", np.array)([2]))
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(b, cast("Callable[..., EagerTensorProtocol]", np.array)([1]))
        (a, b) = cast("Callable[..., EagerTensorProtocol]", execute)("ApproxMaxK", cast("Callable[..., EagerTensorProtocol]", np.array)([]))
        assert cast("EagerTensorProtocol", getattr(a, "size")) == 0
        assert cast("EagerTensorProtocol", getattr(b, "size")) == 0
        c = execute("ForiLoop", None, None, np.array([3]))
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(c, cast("Callable[..., EagerTensorProtocol]", np.array)([3]))
        c = execute("ForiLoop", None)
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(c, cast("Callable[..., EagerTensorProtocol]", np.array)(0))
        d = execute("IgammaGradA", np.array([4]))
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(d, cast("Callable[..., EagerTensorProtocol]", np.array)([4]))
        d = execute("IgammaGradA")
        assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(d, cast("Callable[..., EagerTensorProtocol]", np.array)(0))
