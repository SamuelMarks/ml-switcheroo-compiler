"""JIT compilation wrapper, `disable_jit` context manager, and `eval_shape` utility."""

from __future__ import annotations

import contextlib
import typing

import pydantic

from ml_switcheroo_compiler.core.shape import IRAttrValue, OpValue, VariadicCallableProtocol, is_variadic_callable

from .utils import DifferentiableFn, PrimalTree, as_primal_tree


class JitOptions(pydantic.BaseModel):
    """Configuration options for JIT compilation."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    static_argnums: IRAttrValue = None
    static_argnames: IRAttrValue = None
    donate_argnums: IRAttrValue = None
    donate_argnames: IRAttrValue = None
    keep_unused: bool = False
    device: IRAttrValue = None
    backend: IRAttrValue = None
    inline: bool = False
    abstracted_axes: IRAttrValue = None


def jit(fun: DifferentiableFn, options: JitOptions | None = None) -> VariadicCallableProtocol:
    """Wrap `fun` for just-in-time compilation with `options`.

    Args:
        fun (DifferentiableFn): Function to JIT compile.
        options (JitOptions | None): Optional `JitOptions` configuration.

    Returns:
        VariadicCallableProtocol: JIT-wrapped callable.
    """
    _ = options if options is not None else JitOptions()

    def wrapped(
        *args: OpValue,
        **kwargs: OpValue,
    ) -> PrimalTree:
        """Execute JIT-wrapped function `fun`.

        Args:
            *args (OpValue): Positional call arguments.
            **kwargs (OpValue): Keyword call arguments.

        Returns:
            PrimalTree: Result of evaluating `fun(*args, **kwargs)`.
        """
        _callable_fn_1: OpValue = fun
        assert is_variadic_callable(_callable_fn_1)
        return as_primal_tree(_callable_fn_1(*args, **kwargs))

    return wrapped


@contextlib.contextmanager
def disable_jit() -> typing.Iterator[None]:
    """Context manager that temporarily disables JIT compilation within its scope.

    Yields:
        None: Context scope with JIT disabled.
    """
    yield


def eval_shape(
    fun: DifferentiableFn,
    *args: PrimalTree | OpValue,
    **kwargs: PrimalTree | OpValue,
) -> PrimalTree:
    """Evaluate the output shape and dtype of `fun(*args, **kwargs)`.

    Args:
        fun (DifferentiableFn): Function whose output shape is to be inferred.
        *args (PrimalTree | OpValue): Positional arguments or shape specs.
        **kwargs (PrimalTree | OpValue): Keyword arguments.

    Returns:
        PrimalTree: Evaluated output of `fun(*args, **kwargs)`.
    """
    _callable_fn_2: OpValue = fun
    assert is_variadic_callable(_callable_fn_2)
    return as_primal_tree(_callable_fn_2(*args, **kwargs))
