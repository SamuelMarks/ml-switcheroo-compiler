# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_internal module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Copysign")
def _copysign(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _copysign operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "copysign", None)
    if func:
        return func(*args, **kwargs)
    (x, y) = (args[0], args[1])
    return backend_module.abs(x) * backend_module.sign(y)


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _getprintoptions operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@global_eager_registry.register("TensorArrayRead")
def _np_tensorarrayread(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarrayread operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tensorarrayread", getattr(backend_module, "tensorarrayread", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0][args[1]]


@global_eager_registry.register("TensorArrayWrite")
def _np_tensorarraywrite(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarraywrite operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tensorarraywrite", getattr(backend_module, "tensorarraywrite", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    args[0][args[1]] = args[2]
    return args[0]


@global_eager_registry.register("TopK")
def _np_topk(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_topk operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "topk", getattr(backend_module, "topk", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    idx = np.argsort(args[0])[-args[1] :]
    return (args[0][idx], idx)
