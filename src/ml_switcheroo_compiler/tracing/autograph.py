"""Graph Compilation and AST Processing for AutoGraph capabilities."""

from __future__ import annotations

import functools  # noqa: F401
import typing
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.shape import OpValue, is_variadic_callable
from ml_switcheroo_compiler.tracing.state import global_tracing_state

F = typing.TypeVar("F", bound=OpValue)


class DoNotConvertWrapper(typing.Generic[F]):
    """Typed callable wrapper marking a function to skip AutoGraph AST conversion."""

    def __init__(self, fn: F) -> None:
        """Initialize wrapper around target callable ``fn``.

        Args:
            fn (F): Function excluded from AutoGraph AST conversion.
        """
        self._fn: F = fn
        self._autograph_do_not_convert: bool = True
        self.__wrapped__: DoNotConvertWrapper[F] = self

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Invoke wrapped callable unchanged.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            OpValue: Output from wrapped callable.
        """
        _callable_fn_1: OpValue = self._fn
        assert is_variadic_callable(_callable_fn_1)
        return _callable_fn_1(*args, **kwargs)


def _set_graph_loop_options(active_g: LogicalGraph, opts: LoopOptions | None) -> None:
    """Attach ``opts`` to ``active_g`` and its ``metadata`` dictionary.

    Args:
        active_g (LogicalGraph): Active graph instance.
        opts (LoopOptions | None): Loop options to attach.
    """
    active_g.loop_options = opts
    active_g.metadata["loop_options"] = opts


ShapeInvariantType = typing.Union[
    tuple[typing.Union[int, "None"], ...],
    Sequence[tuple[typing.Union[int, "None"], ...]],
    Mapping[str, tuple[typing.Union[int, "None"], ...]],
    Sequence[typing.Union[int, "None"]],
    tuple[Sequence[typing.Union[int, "None"]], ...],
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
            parallel_iterations (int | None): Maximum number of iterations allowed to run in parallel.
            swap_memory (bool | None): Whether to enable CPU-GPU memory swapping for large loops.
            maximum_iterations (int | None): Maximum number of loop iterations to execute.
            shape_invariants (ShapeInvariantType | None): Metadata describing invariant shape constraints.
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
            _set_graph_loop_options(global_tracing_state.active_graph, self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit loop options context and restore previous tracing state.

        Args:
            exc_type (type[BaseException] | None): Exception type if an exception was raised.
            exc_val (BaseException | None): Exception instance if an exception was raised.
            exc_tb (TracebackType | None): Traceback if an exception was raised.
        """
        global_tracing_state.current_loop_options = self._prev_options
        if global_tracing_state.active_graph is not None:
            _set_graph_loop_options(global_tracing_state.active_graph, self._prev_options)


def set_loop_options(
    parallel_iterations: int | None = None,
    swap_memory: bool | None = None,
    maximum_iterations: int | None = None,
    shape_invariants: ShapeInvariantType | None = None,
) -> LoopOptions:
    """Set dynamic loop unrolling options for a loop in the active graph trace.

    Args:
        parallel_iterations (int | None): Maximum number of parallel loop iterations.
        swap_memory (bool | None): Enable or disable memory swapping.
        maximum_iterations (int | None): Hard limit on the number of loop iterations.
        shape_invariants (ShapeInvariantType | None): Tensor shape invariants to enforce during loop execution.

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
        _set_graph_loop_options(global_tracing_state.active_graph, options)
    return options


@typing.overload
def do_not_convert(func: F) -> DoNotConvertWrapper[F]: ...


@typing.overload
def do_not_convert(func: None = None) -> Callable[[F], DoNotConvertWrapper[F]]: ...


def do_not_convert(
    func: F | None = None,
) -> DoNotConvertWrapper[F] | Callable[[F], DoNotConvertWrapper[F]]:
    """Decorate a callable to prevent AutoGraph from converting it into a traced graph.

    Args:
        func (F | None): Optional callable to wrap directly.

    Returns:
        DoNotConvertWrapper[F] | Callable[[F], DoNotConvertWrapper[F]]:
            Wrapped callable or decorator factory.
    """

    def decorator(f: F) -> DoNotConvertWrapper[F]:
        """Wrap ``f`` in a ``DoNotConvertWrapper``.

        Args:
            f (F): Target callable.

        Returns:
            DoNotConvertWrapper[F]: Wrapped callable marked with ``_autograph_do_not_convert``.
        """
        return DoNotConvertWrapper(f)

    if func is not None:
        return decorator(func)
    return decorator
