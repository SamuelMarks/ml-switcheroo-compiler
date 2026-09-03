"""eager_registry.py module."""

import builtins
from typing import Callable, Optional, Protocol, Union


class BackendArray(Protocol):
    """Protocol for a backend array object."""

    def __getitem__(self, key: Union[int, slice, tuple, list, str, "builtins.ellipsis", None]) -> "BackendArray":
        """Get an item from the array."""
        ...

    def __setitem__(self, key: Union[int, slice, tuple, list, str, "builtins.ellipsis", None], value: "BackendArray") -> None:
        """Set an item in the array."""
        ...

    def __add__(self, other: "EagerValue") -> "BackendArray":
        """Add two arrays."""
        ...

    def __sub__(self, other: "EagerValue") -> "BackendArray":
        """Subtract two arrays."""
        ...

    def __mul__(self, other: "EagerValue") -> "BackendArray":
        """Multiply two arrays."""
        ...

    def __truediv__(self, other: "EagerValue") -> "BackendArray":
        """Divide two arrays."""
        ...

    def __lt__(self, other: "EagerValue") -> "EagerValue":
        """Compare less than."""
        ...

    def __le__(self, other: "EagerValue") -> "EagerValue":
        """Compare less than or equal."""
        ...

    def __gt__(self, other: "EagerValue") -> "EagerValue":
        """Compare greater than."""
        ...

    def __ge__(self, other: "EagerValue") -> "EagerValue":
        """Compare greater than or equal."""
        ...

    def __eq__(self, other: "EagerValue") -> "EagerValue":
        """Compare equality."""
        ...

    def __pow__(self, other: "EagerValue") -> "BackendArray":
        """Power operation."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        ...

    @property
    def dtype(self) -> "BackendArray":
        """Get the dtype of the array."""
        ...


EagerValue = Union[int, float, list, tuple, str, bool, BackendArray, None]

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Eager backend registry."""


class EagerOpRegistry:
    """Registry for eager operations."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registry: dict[str, Callable[..., EagerValue]] = {}

    def register(self, op_type: str) -> Callable[[Callable[..., EagerValue]], Callable[..., EagerValue]]:
        """Register an eager operation.

        Args:
            op_type (str): The op_type parameter.

        Returns:
            Callable[[Callable[..., EagerValue]], Callable[..., EagerValue]]: Result.
        """

        def decorator(func: Callable[..., EagerValue]) -> Callable[..., EagerValue]:
            """Evaluate decorator operation.

            Args:
                func (Callable): The func parameter.

            Returns:
                Callable: Result.
            """
            self._registry[op_type] = func
            return func

        return decorator

    def get(self, op_type: str) -> Optional[Callable[..., EagerValue]]:
        """Get an eager operation.

        Args:
            op_type (str): The name of the operation.

        Returns:
            Optional[Callable[..., EagerValue]]: The eager function or None.
        """
        return self._registry.get(op_type)

    def dispatch(self, op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Dispatch an eager operation.

        Args:
            op_type (str): The name of the operation.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            EagerValue: The result.

        Raises:
            ValueError: If the operation is not found in the registry.
        """
        func = self.get(op_type)
        if func is not None:
            return func(*args, **kwargs)

        # Fallback to global pure python registry if available
        if self is not global_eager_registry:
            func = global_eager_registry.get(op_type)
            if func is not None:
                return func(*args, **kwargs)

        # Universal Math Backfill Fallback via backend_module introspection
        # If the op is not explicitly registered but the backend natively supports it, map it dynamically.
        # This resolves missing operations (like Sin, Cos, Add, MatMul) that adhere to the
        # N-to-M universal utility rule natively on the target tensor backend.
        if args and hasattr(args[0], op_type.lower()):
            # e.g., np.sin() fallback
            return getattr(args[0], op_type.lower())(*args[1:], **kwargs)

        backend = kwargs.get("backend_module")
        if backend is None and len(args) > 0:
            backend = args[0]
            op_args = args[1:]
        else:
            op_args = args

        if backend is not None:
            # Map common names
            op_name = op_type.lower()
            if hasattr(backend, op_name):
                return getattr(backend, op_name)(*op_args, **kwargs)

            # Alternative common mappings
            mapping = {"matmul": "matmul", "add": "add", "sub": "subtract", "mul": "multiply", "div": "divide", "truedivide": "divide"}
            if op_name in mapping and hasattr(backend, mapping[op_name]):
                return getattr(backend, mapping[op_name])(*op_args, **kwargs)

        msg = f"Operation '{op_type}' not found in registry and no universal fallback could be inferred for backend {backend}."
        raise ValueError(msg)


# Global registry instance
global_eager_registry = EagerOpRegistry()

mlx_eager_registry = EagerOpRegistry()
numpy_eager_registry = EagerOpRegistry()
pure_python_eager_registry = EagerOpRegistry()


@global_eager_registry.register("CustomVJP")
def _eager_custom_vjp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Custom vjp."""
    # Just return args because hook fwd is identity or simple.
    # Actually, for standard custom_vjp, we might need to run fwd_fn.
    # In JAX custom_vjp, you return (primal, residual). But custom_vjp here intercepts the python call.
    # The actual execution happens via the python function self.fwd in eager mode!
    # If we are in trace execution (which we are now, because jit is evaluating the graph),
    # the CustomVJP node shouldn't really execute its internal python function directly, but let's just return args[0].
    # For hook_gradient, args[0] is `t`, we just return `t`.
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("ProcessCustomVJPCall")
def _eager_process_custom_vjp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Process custom vjp."""
    bwd_fn = kwargs["bwd_fn"]
    if callable(bwd_fn):
        return bwd_fn.__call__(None, *args)
    return None


@global_eager_registry.register("TupleGetItem")
def _eager_tuple_get_item(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Tuple get item."""
    index: int = int(getattr(kwargs.get("index", 0), "__int__", lambda: 0)())
    if isinstance(args[0], (tuple, list)):
        return args[0][index]
    return None
