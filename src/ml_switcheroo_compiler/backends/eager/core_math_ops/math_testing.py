# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_testing module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Allclose")
def _allclose(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _allclose operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = args[0]
    b = args[1]
    rtol = kwargs.get("rtol", 1e-05)
    atol = kwargs.get("atol", 1e-08)
    equal_nan = kwargs.get("equal_nan", False)

    def _val(x: Any) -> Any:
        """Evaluate _val operation.

        Args:
        x (object): The x parameter.

        Returns: Any: Result.
        """
        x_data = getattr(x, "data", x)
        if hasattr(x_data, "item") and callable(x_data.item):
            return x_data.item()
        if hasattr(x_data, "tolist"):
            return x_data.tolist()
        return x_data

    if hasattr(backend_module, "allclose"):
        return backend_module.allclose(a, b, rtol=float(_val(rtol)), atol=float(_val(atol)), equal_nan=bool(_val(equal_nan)))
    return None


@global_eager_registry.register("ArrayEquiv")
def _array_equiv(backend_module: Any, a1: Any, a2: Any, **kwargs: Any) -> Any:
    """Evaluate _array_equiv operation.

    Args:
        backend_module (object): The backend_module parameter.
        a1 (object): The a1 parameter.
        a2 (object): The a2 parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.allclose(a1, a2) if hasattr(backend_module, "allclose") else True


@global_eager_registry.register("Assert")
def _assert(backend_module: Any, condition: Any, data: Any, summarize: int = 3, **kwargs: Any) -> Any:
    """Evaluate _assert operation.

    Args:
        backend_module (object): The backend_module parameter.
        condition (object): The condition parameter.
        data (object): The data parameter.
        summarize (int): The summarize parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return None


@global_eager_registry.register("PromoteTypes")
def _promotetypes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _promotetypes operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.promote_types(*args, **kwargs)


@global_eager_registry.register("ResultType")
def _resulttype(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _resulttype operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.result_type(*args, **kwargs)
