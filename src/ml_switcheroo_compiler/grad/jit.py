"""JIT compilation wrapper, `disable_jit` context manager, and `eval_shape` utility."""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from ml_switcheroo_compiler.core.shape import AttrValue

from .utils import GradValue


@dataclass
class JitOptions:
    """Configuration options for JIT compilation."""

    static_argnums: AttrValue = None
    static_argnames: AttrValue = None
    donate_argnums: AttrValue = None
    donate_argnames: AttrValue = None
    keep_unused: bool = False
    device: AttrValue = None
    backend: AttrValue = None
    inline: bool = False
    abstracted_axes: AttrValue = None


def jit(fun: Callable[..., GradValue], options: JitOptions | GradValue = None) -> Callable[..., GradValue]:
    """Wrap `fun` for just-in-time compilation with `options`.

    Args:
        fun (Callable[..., GradValue]): Function to JIT compile.
        options (JitOptions | GradValue): Optional `JitOptions` configuration.

    Returns:
        Callable[..., GradValue]: JIT-wrapped callable.
    """
    _ = cast(GradValue, options or JitOptions())

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Execute JIT-wrapped function `fun`.

        Args:
            *args (GradValue): Positional call arguments.
            **kwargs (GradValue): Keyword call arguments.

        Returns:
            GradValue: Result of evaluating `fun(*args, **kwargs)`.
        """
        return fun(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def disable_jit() -> typing.Iterator[None]:
    """Context manager that temporarily disables JIT compilation within its scope.

    Yields:
        None: Context scope with JIT disabled.
    """
    yield


def eval_shape(fun: Callable[..., GradValue], *args: GradValue, **kwargs: GradValue) -> GradValue:
    """Evaluate the output shape and dtype of `fun(*args, **kwargs)`.

    Args:
        fun (Callable[..., GradValue]): Function whose output shape is to be inferred.
        *args (GradValue): Positional arguments or shape specs.
        **kwargs (GradValue): Keyword arguments.

    Returns:
        GradValue: Evaluated output of `fun(*args, **kwargs)`.
    """
    return fun(*args, **kwargs)
