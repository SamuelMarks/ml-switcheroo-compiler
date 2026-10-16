"""math_polynomial module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Polyval")
def _polyval(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _polyval operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "polyval", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Polyint")
def _np_polyint(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_polyint operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "polyint", getattr(backend_module, "polyint", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.polyint(*args, **kwargs))
