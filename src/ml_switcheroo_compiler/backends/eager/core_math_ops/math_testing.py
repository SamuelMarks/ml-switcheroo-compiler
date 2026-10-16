"""math_testing module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Allclose")
def _allclose(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _allclose operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    a = args[0]
    b = args[1]
    rtol = kwargs.get("rtol", 1e-05) if hasattr(kwargs, "get") else 1e-05
    atol = kwargs.get("atol", 1e-08) if hasattr(kwargs, "get") else 1e-08
    equal_nan = kwargs.get("equal_nan", False) if hasattr(kwargs, "get") else False

    def _val(x: EagerValue) -> EagerValue:
        """Evaluate _val operation.

        Args:
        x: The x parameter.

        Returns:
            object: Result.
        """
        x_data = getattr(x, "data", x)
        if hasattr(x_data, "item") and callable(getattr(x_data, "item")):
            return cast(EagerValue, getattr(x_data, "item")())
        if hasattr(x_data, "tolist"):
            return cast(EagerValue, getattr(x_data, "tolist")())
        return x_data

    if hasattr(backend_module, "allclose"):
        return cast(EagerValue, getattr(backend_module, "allclose")(a, b, rtol=float(cast(float, _val(rtol))), atol=float(cast(float, _val(atol))), equal_nan=bool(_val(equal_nan))))
    return cast(EagerValue, np.allclose(a, b, rtol=float(cast(float, _val(rtol))), atol=float(cast(float, _val(atol))), equal_nan=bool(_val(equal_nan))))


@global_eager_registry.register("ArrayEquiv")
def _array_equiv(backend_module: EagerValue, a1: EagerValue, a2: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _array_equiv operation.

    Args:
        backend_module: The backend_module parameter.
        a1: The a1 parameter.
        a2: The a2 parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return getattr(backend_module, "allclose")(a1, a2) if hasattr(backend_module, "allclose") else True


@global_eager_registry.register("Assert")
def _assert(backend_module: EagerValue, condition: EagerValue, data: EagerValue = None, summarize: int = 3, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _assert operation.

    Args:
        backend_module: The backend_module parameter.
        condition: The condition parameter.
        data: Optional tensor data or sequence to summarize upon assertion failure.
        summarize (int): The summarize parameter.
        **kwargs: Keyword args (e.g. node_id, message).

    Returns:
        None: When assertion succeeds.

    Raises:
        AssertionError: When condition evaluates to False.
    """
    cond_arr = np.asarray(condition)
    if not np.all(cond_arr):
        node_id = kwargs.get("node_id", "AssertOp")
        msg = kwargs.get("message", "Condition evaluated to False")
        if data is not None:
            data_arr = np.asarray(data)
            flat = data_arr.reshape(-1)
            summary_data = flat[:summarize].tolist() if len(flat) > 0 else []
            raise AssertionError(f"Assertion failed in node '{node_id}': {msg}. Data sample: {summary_data}")
        raise AssertionError(f"Assertion failed in node '{node_id}': {msg}")
    return None


@global_eager_registry.register("PromoteTypes")
def _promotetypes(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _promotetypes operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "promote_types")(*args, **kwargs))


@global_eager_registry.register("ResultType")
def _resulttype(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _resulttype operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, getattr(backend_module, "result_type")(*args, **kwargs))
