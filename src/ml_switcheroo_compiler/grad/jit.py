"""JIT compilation wrapper, `disable_jit` context manager, and `eval_shape` utility."""

from __future__ import annotations

import contextlib
import typing

import pydantic

from ml_switcheroo_compiler.core.shape import AttrValue, VariadicCallableProtocol, invoke_callable

from .utils import GradValue


class JitOptions(pydantic.BaseModel):
    """Configuration options for JIT compilation."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    static_argnums: AttrValue = None
    static_argnames: AttrValue = None
    donate_argnums: AttrValue = None
    donate_argnames: AttrValue = None
    keep_unused: bool = False
    device: AttrValue = None
    backend: AttrValue = None
    inline: bool = False
    abstracted_axes: AttrValue = None


def jit(fun: GradValue, options: JitOptions | GradValue = None) -> VariadicCallableProtocol:
    """Wrap `fun` for just-in-time compilation with `options`.

    Args:
        fun (GradValue): Function to JIT compile.
        options (JitOptions | GradValue): Optional `JitOptions` configuration.

    Returns:
        VariadicCallableProtocol: JIT-wrapped callable.
    """
    _ = options if options is not None else JitOptions()

    def wrapped(*args: GradValue, **kwargs: GradValue) -> GradValue:
        """Execute JIT-wrapped function `fun`.

        Args:
            *args (GradValue): Positional call arguments.
            **kwargs (GradValue): Keyword call arguments.

        Returns:
            GradValue: Result of evaluating `fun(*args, **kwargs)`.
        """
        return invoke_callable(fun, *args, **kwargs)

    return wrapped


@contextlib.contextmanager
def disable_jit() -> typing.Iterator[None]:
    """Context manager that temporarily disables JIT compilation within its scope.

    Yields:
        None: Context scope with JIT disabled.
    """
    yield


def eval_shape(fun: GradValue, *args: GradValue, **kwargs: GradValue) -> GradValue:
    """Evaluate the output shape and dtype of `fun(*args, **kwargs)`.

    Args:
        fun (GradValue): Function whose output shape is to be inferred.
        *args (GradValue): Positional arguments or shape specs.
        **kwargs (GradValue): Keyword arguments.

    Returns:
        GradValue: Evaluated output of `fun(*args, **kwargs)`.
    """
    return invoke_callable(fun, *args, **kwargs)
