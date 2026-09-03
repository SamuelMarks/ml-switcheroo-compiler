# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_internal module."""

from __future__ import annotations

import builtins
import typing
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Copysign")
def _copysign(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _copysign operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "copysign", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    y = args[1]
    return backend_module.abs(x) * backend_module.sign(y)


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _getprintoptions operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@global_eager_registry.register("NpTensorarrayread")
def _np_tensorarrayread(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarrayread operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return 0


@global_eager_registry.register("NpTensorarraywrite")
def _np_tensorarraywrite(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarraywrite operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    arr, index, value = args[0], args[1], args[2]
    if isinstance(arr, list):
        res = list(arr)
        res[int(index)] = value
        return res
    return 0


@global_eager_registry.register("NpTopk")
def _np_topk(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_topk operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    arr = args[0]
    k = args[1] if len(args) > 1 else 1
    import numpy as np

    if isinstance(arr, (list, np.ndarray)):
        arr_np = np.array(arr)
        idx = np.argsort(arr_np)[-k:][::-1]
        return arr_np[idx], idx
    return [0], [0]
