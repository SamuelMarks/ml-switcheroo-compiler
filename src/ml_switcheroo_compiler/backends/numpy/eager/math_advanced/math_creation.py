# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_creation module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Fromfunction")
def _np_fromfunction_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fromfunction via fromfunction.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fromfunction(*args, **kwargs)


@numpy_eager_registry.register("Fromiter")
def _np_fromiter_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fromiter via fromiter.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fromiter(*args, **kwargs)


@numpy_eager_registry.register("Frompyfunc")
def _np_frompyfunc_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Frompyfunc via frompyfunc.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.frompyfunc(*args, **kwargs)


@numpy_eager_registry.register("Geomspace")
def _np_geomspace_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Geomspace via geomspace.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.geomspace(*args, **kwargs)


@numpy_eager_registry.register("Zeros")
def _np_zeros_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Zeros via zeros.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.zeros(*args, **kwargs)


@numpy_eager_registry.register("Ones")
def _np_ones_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Ones via ones.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ones(*args, **kwargs)


@numpy_eager_registry.register("Empty")
def _np_empty_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Empty via empty.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.empty(*args, **kwargs)


@numpy_eager_registry.register("Full")
def _np_full_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Full via full.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.full(*args, **kwargs)


@numpy_eager_registry.register("ZerosLike")
def _np_zeros_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ZerosLike via zeros_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.zeros_like(*args, **kwargs)


@numpy_eager_registry.register("OnesLike")
def _np_ones_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement OnesLike via ones_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ones_like(*args, **kwargs)


@numpy_eager_registry.register("EmptyLike")
def _np_empty_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement EmptyLike via empty_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.empty_like(*args, **kwargs)


@numpy_eager_registry.register("FullLike")
def _np_full_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement FullLike via full_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.full_like(*args, **kwargs)


@numpy_eager_registry.register("Arange")
def _np_arange_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Arange via arange.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.arange(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorIdentity")
def _np_linearoperatoridentity(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorIdentity.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorIdentity

    return LinearOperatorIdentity(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorScaledIdentity")
def _np_linearoperatorscaledidentity(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorScaledIdentity.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorScaledIdentity

    return LinearOperatorScaledIdentity(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorZeros")
def _np_linearoperatorzeros(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorZeros.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorZeros

    return LinearOperatorZeros(*args, **kwargs)


@numpy_eager_registry.register("FromDlpack")
def _np_fromdlpack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fromdlpack operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "from_dlpack"):
        return backend_module.from_dlpack(*args, **kwargs)
    return args[0]


@numpy_eager_registry.register("Frombuffer")
def _np_frombuffer(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_frombuffer operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    if not args:
        return None
    return np.frombuffer(args[0], **kwargs)
