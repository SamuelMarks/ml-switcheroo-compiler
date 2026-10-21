"""Implementation of math testing operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_float, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Allclose")
def _allclose(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager allclose operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    a = args[0]
    b = args[1]
    rtol = kwargs.get("rtol", 1e-05) if hasattr(kwargs, "get") else 1e-05
    atol = kwargs.get("atol", 1e-08) if hasattr(kwargs, "get") else 1e-08
    equal_nan = kwargs.get("equal_nan", False) if hasattr(kwargs, "get") else False

    def _val(x: EagerValue) -> EagerValue:
        """Execute the eager val operation on the active backend array.

        Args:
        x: The x parameter.

        Returns:
            EagerValue: Result.
        """
        x_data = getattr(x, "data", x)
        if hasattr(x_data, "item") and callable(getattr(x_data, "item")):
            return as_tensor(getattr(x_data, "item")())
        if hasattr(x_data, "tolist"):
            return as_tensor(getattr(x_data, "tolist")())
        return x_data

    if hasattr(backend_module, "allclose"):
        return as_tensor(getattr(backend_module, "allclose")(a, b, rtol=float(as_float(_val(rtol))), atol=float(as_float(_val(atol))), equal_nan=bool(_val(equal_nan))))
    return _resolve_eager_mod(backend_module).allclose(a, b, rtol=float(as_float(_val(rtol))), atol=float(as_float(_val(atol))), equal_nan=bool(_val(equal_nan)))


@global_eager_registry.register("ArrayEquiv")
def _array_equiv(backend_module: BackendModuleProtocol, a1: EagerValue, a2: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager array equiv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        a1: The a1 parameter.
        a2: The a2 parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return getattr(backend_module, "allclose")(a1, a2) if hasattr(backend_module, "allclose") else True


@global_eager_registry.register("Assert")
def _assert(backend_module: BackendModuleProtocol, condition: EagerValue, data: EagerValue = None, summarize: int = 3, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager assert operation on the active backend array.

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
    mod = _resolve_eager_mod(backend_module)
    cond_arr = mod.asarray(condition)
    if not mod.all(cond_arr):
        node_id = kwargs.get("node_id", "AssertOp")
        msg = kwargs.get("message", "Condition evaluated to False")
        if data is not None:
            data_arr = mod.asarray(data)
            flat = data_arr.reshape(-1)
            summary_data = flat[:summarize].tolist() if len(flat) > 0 else []
            raise AssertionError(f"Assertion failed in node '{node_id}': {msg}. Data sample: {summary_data}")
        raise AssertionError(f"Assertion failed in node '{node_id}': {msg}")
    return None


@global_eager_registry.register("PromoteTypes")
def _promotetypes(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager promotetypes operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "promote_types")(*args, **kwargs))


@global_eager_registry.register("ResultType")
def _resulttype(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager resulttype operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "result_type")(*args, **kwargs))
