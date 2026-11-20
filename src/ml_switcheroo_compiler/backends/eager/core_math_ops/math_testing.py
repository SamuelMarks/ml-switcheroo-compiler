"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Allclose")
def _allclose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager allclose operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    a = args[0]
    b = args[1]
    rtol = kwargs.get("rtol", 1e-05) if adapt_backend_module(kwargs).has_symbol("get") else 1e-05
    atol = kwargs.get("atol", 1e-08) if adapt_backend_module(kwargs).has_symbol("get") else 1e-08
    equal_nan = kwargs.get("equal_nan", False) if adapt_backend_module(kwargs).has_symbol("get") else False

    def _val(x: BackendArg) -> BackendArg:
        """Execute the eager val operation on the active backend array.

        Args:
        x: Input `x` argument (x).

        Returns:
            BackendArg: Computed `BackendArg` output.
        """
        x_data = adapt_backend_module(x).get_symbol("data", x)
        if adapt_backend_module(x_data).has_symbol("item") and callable(
            adapt_backend_module(x_data).require_symbol("item")
        ):
            return adapt_backend_module(x_data).require_symbol("item")()
        if adapt_backend_module(x_data).has_symbol("tolist"):
            return adapt_backend_module(x_data).require_symbol("tolist")()
        _typed_ret_1: BackendArg = x_data
        return _typed_ret_1

    if adapt_backend_module(backend_module).has_symbol("allclose"):
        return adapt_backend_module(backend_module).require_symbol("allclose")(
            a,
            b,
            rtol=float(coerce_float(_val(rtol))),
            atol=float(coerce_float(_val(atol))),
            equal_nan=bool(_val(equal_nan)),
        )
    return _resolve_eager_mod(backend_module).allclose(
        a,
        b,
        rtol=float(coerce_float(_val(rtol))),
        atol=float(coerce_float(_val(atol))),
        equal_nan=bool(_val(equal_nan)),
    )


@global_eager_registry.register("ArrayEquiv")
def _array_equiv(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    a1: BackendArg,
    a2: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager array equiv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        a1: Input `a1` argument (a1).
        a2: Input `a2` argument (a2).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return (
        adapt_backend_module(backend_module).require_symbol("allclose")(a1, a2)
        if adapt_backend_module(backend_module).has_symbol("allclose")
        else True
    )


@global_eager_registry.register("Assert")
def _assert(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    condition: BackendArg,
    data: BackendArg = None,
    summarize: int = 3,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager assert operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        condition: Input `condition` argument (condition).
        data: Optional tensor data or sequence to summarize upon assertion failure.
        summarize (int): Input `summarize` argument (summarize).
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
def _promotetypes(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager promotetypes operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("promote_types")(*args, **kwargs)


@global_eager_registry.register("ResultType")
def _resulttype(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager resulttype operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("result_type")(*args, **kwargs)
