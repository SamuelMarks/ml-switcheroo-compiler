# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Graph Compilation and AST Processing for AutoGraph capabilities."""

import functools
from typing import Callable, Optional, TypeVar, Union

F = TypeVar("F", bound=Callable[..., object])


class LoopOptions:
    """Dynamic loop unrolling options in the graph lowerer."""

    def __init__(
        self,
        parallel_iterations: Optional[int] = None,
        swap_memory: Optional[bool] = None,
        maximum_iterations: Optional[int] = None,
        shape_invariants: Optional[object] = None,
    ) -> None:
        """Initialize the LoopOptions.

        Args:
            parallel_iterations (Optional[int]): The maximum number of iterations allowed to run in parallel.
            swap_memory (Optional[bool]): Whether to enable CPU-GPU memory swapping for large loops.
            maximum_iterations (Optional[int]): The maximum number of loop iterations to execute.
            shape_invariants (Optional[object]): Metadata describing the invariant shape constraints of loop variables.
        """
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory
        self.maximum_iterations = maximum_iterations
        self.shape_invariants = shape_invariants


def set_loop_options(
    parallel_iterations: Optional[int] = None,
    swap_memory: Optional[bool] = None,
    maximum_iterations: Optional[int] = None,
    shape_invariants: Optional[object] = None,
) -> None:
    """Set dynamic loop unrolling options for a loop in the active graph trace.

    Args:
        parallel_iterations (Optional[int]): The maximum number of parallel loop iterations.
        swap_memory (Optional[bool]): Enable or disable memory swapping.
        maximum_iterations (Optional[int]): Hard limit on the number of loop iterations.
        shape_invariants (Optional[object]): Tensor shape invariants to enforce during the loop execution.
    """
    # In a full implementation, this might push a context onto the builder
    # No-op in eager trace
    return


def do_not_convert(func: Optional[F] = None) -> Union[F, Callable[[F], F]]:
    """Decorate to prevent AutoGraph from converting a function into a graph representation.

    Args:
        func (Optional[F]): The function to exclude from AST conversion.

    Returns:
        Union[F, Callable[[F], F]]: The original function, modified to skip AutoGraph processing.
    """

    def decorator(f: F) -> F:
        """Mark a function to skip AutoGraph conversion.

        Args:
            f (F): The inner function being wrapped by the decorator.

        Returns:
            F: The original function with the skip flag attached.
        """
        f._autograph_do_not_convert = True

        @functools.wraps(f)
        def wrapper(*args: object, **kwargs: object) -> object:
            """Execute the uncoverted function directly.

            Args:
                *args (object): Positional arguments passed to the function.
                **kwargs (object): Keyword arguments passed to the function.

            Returns:
                object: The return value of the wrapped function.
            """
            return f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
