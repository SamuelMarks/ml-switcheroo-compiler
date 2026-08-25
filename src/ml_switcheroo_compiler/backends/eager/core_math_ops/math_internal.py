# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_internal module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Copysign")
def _copysign(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _copysign operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "copysign", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    y: typing.Any = args[1]
    return backend_module.abs(x) * backend_module.sign(y)


@global_eager_registry.register("GetPrintoptions")
def _getprintoptions(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _getprintoptions operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@global_eager_registry.register("NpTensorarrayread")
def _np_tensorarrayread(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _np_tensorarrayread operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return 0


@global_eager_registry.register("NpTensorarraywrite")
def _np_tensorarraywrite(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _np_tensorarraywrite operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return 0


@global_eager_registry.register("NpTopk")
def _np_topk(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _np_topk operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return 0
