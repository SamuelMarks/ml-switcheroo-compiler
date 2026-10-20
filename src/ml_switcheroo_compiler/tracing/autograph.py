"""Graph Compilation and AST Processing for AutoGraph capabilities."""

from __future__ import annotations

import functools  # noqa: F401
import typing
from collections.abc import Callable
from types import TracebackType

from ml_switcheroo_compiler.tracing.state import global_tracing_state

F = typing.TypeVar("F")

ShapeInvariantType = typing.Union[
    tuple[typing.Union[int, "None"], ...],
    list[tuple[typing.Union[int, "None"], ...]],
    dict[str, tuple[typing.Union[int, "None"], ...]],
    list[typing.Union[int, "None"]],
    tuple[list[typing.Union[int, "None"]], ...],
    "None",
]


class LoopOptions:
    """Dynamic loop unrolling options in the graph lowerer."""

    def __init__(
        self,
        parallel_iterations: int | None = None,
        swap_memory: bool | None = None,
        maximum_iterations: int | None = None,
        shape_invariants: ShapeInvariantType | None = None,
    ) -> None:
        """Initialize the LoopOptions.

        Args:
            parallel_iterations (Optional[int]): The maximum number of iterations allowed to run in parallel.
            swap_memory (Optional[bool]): Whether to enable CPU-GPU memory swapping for large loops.
            maximum_iterations (Optional[int]): The maximum number of loop iterations to execute.
            shape_invariants (Optional[ShapeInvariantType]): Metadata describing the invariant shape constraints of loop variables.
        """
        self.parallel_iterations: int | None = parallel_iterations
        self.swap_memory: bool | None = swap_memory
        self.maximum_iterations: int | None = maximum_iterations
        self.shape_invariants: ShapeInvariantType | None = shape_invariants
        self._prev_options: LoopOptions | None = None

    def __enter__(self) -> LoopOptions:
        """Enter loop options context and inject onto active tracing state.

        Returns:
            LoopOptions: The active loop options instance.
        """
        self._prev_options = global_tracing_state.current_loop_options
        global_tracing_state.current_loop_options = self
        if global_tracing_state.active_graph is not None:
            object.__setattr__(global_tracing_state.active_graph, "loop_options", self)
            metadata = getattr(global_tracing_state.active_graph, "metadata", None)
            if isinstance(metadata, dict):
                metadata["loop_options"] = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit loop options context and restore previous tracing state.

        Args:
            exc_type (Optional[type[BaseException]]): Exception type if an exception was raised.
            exc_val (Optional[BaseException]): Exception instance if an exception was raised.
            exc_tb (Optional[TracebackType]): Traceback if an exception was raised.
        """
        global_tracing_state.current_loop_options = self._prev_options
        if global_tracing_state.active_graph is not None:
            object.__setattr__(global_tracing_state.active_graph, "loop_options", self._prev_options)
            metadata = getattr(global_tracing_state.active_graph, "metadata", None)
            if isinstance(metadata, dict):
                metadata["loop_options"] = self._prev_options


def set_loop_options(
    parallel_iterations: int | None = None,
    swap_memory: bool | None = None,
    maximum_iterations: int | None = None,
    shape_invariants: ShapeInvariantType | None = None,
) -> LoopOptions:
    """Set dynamic loop unrolling options for a loop in the active graph trace.

    Args:
        parallel_iterations (Optional[int]): The maximum number of parallel loop iterations.
        swap_memory (Optional[bool]): Enable or disable memory swapping.
        maximum_iterations (Optional[int]): Hard limit on the number of loop iterations.
        shape_invariants (Optional[ShapeInvariantType]): Tensor shape invariants to enforce during the loop execution.

    Returns:
        LoopOptions: The configured LoopOptions instance injected into the active trace context.
    """
    options = LoopOptions(
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        maximum_iterations=maximum_iterations,
        shape_invariants=shape_invariants,
    )
    if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
        global_tracing_state.current_loop_options = options
        object.__setattr__(global_tracing_state.active_graph, "loop_options", options)
        metadata = getattr(global_tracing_state.active_graph, "metadata", None)
        if isinstance(metadata, dict):
            metadata["loop_options"] = options
    return options


def do_not_convert(func: F | None = None) -> F | Callable[[F], F]:
    """Decorate to prevent AutoGraph from converting a function into a graph representation.

    Args:
        func (Optional[F]): The function to exclude from AST conversion.

    Returns:
        Union[F, Callable[[F], F]]: The original function, modified to skip AutoGraph processing.
    """

    def decorator(f: F) -> F:
        """Attach the do-not-convert attribute to the function.

        Args:
            f (F): The function to be marked.

        Returns:
            F: The original function with the skip flag attached.
        """
        setattr(f, "_autograph_do_not_convert", True)
        setattr(f, "__wrapped__", f)
        return f

    if func is not None:
        return decorator(func)
    return decorator
