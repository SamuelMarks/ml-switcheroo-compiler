# ruff: noqa: D107, ANN401
"""Graph Compilation and AST Processing for AutoGraph capabilities."""

import functools
from typing import Any, Callable, Optional, TypeVar, Union

F = TypeVar("F", bound=Callable[..., Any])


class LoopOptions:
    """Dynamic loop unrolling options in the graph lowerer."""

    def __init__(
        self,
        parallel_iterations: Optional[int] = None,
        swap_memory: Optional[bool] = None,
        maximum_iterations: Optional[int] = None,
        shape_invariants: Optional[Any] = None,
    ) -> None:
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory
        self.maximum_iterations = maximum_iterations
        self.shape_invariants = shape_invariants


def set_loop_options(
    parallel_iterations: Optional[int] = None,
    swap_memory: Optional[bool] = None,
    maximum_iterations: Optional[int] = None,
    shape_invariants: Optional[Any] = None,
) -> None:
    """Set dynamic loop unrolling options for a loop in the active graph trace.

    This function configures loop options for the compiler backend.
    """
    # In a full implementation, this might push a context onto the builder
    # No-op in eager trace
    return


def do_not_convert(func: Optional[F] = None) -> Union[F, Callable[[F], F]]:
    """Decorator to prevent AutoGraph from converting a function.

    This acts as a tracing bail-out primitive.
    """

    def decorator(f: F) -> F:
        f._autograph_do_not_convert = True

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator
