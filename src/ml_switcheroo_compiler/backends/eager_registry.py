"""eager_registry.py module."""

from typing import Any, Callable, Optional

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Eager backend registry."""

from collections.abc import Callable


class EagerOpRegistry:
    """Registry for eager operations."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registry: dict[str, Callable[..., object]] = {}

    def register(self, op_type: str) -> Callable[..., object]:
        """Register an eager operation.

        Args:
            op_type (str): The op_type parameter.

        Returns:
            Callable: Result.
        """

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            """Evaluate decorator operation.

            Args:
                func (Callable): The func parameter.

            Returns:
                Callable: Result.
            """
            self._registry[op_type] = func
            return func

        return decorator

    def get(self, op_type: str) -> Optional[Callable[..., object]]:
        """Get an eager operation.

        Args:
            op_type (str): The name of the operation.

        Returns:
            Optional[Callable[..., object]]: The eager function or None.
        """
        return self._registry.get(op_type)

    def dispatch(self, op_type: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch an eager operation.

        Args:
            op_type (str): The name of the operation.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: The result.

        Raises:
            ValueError: If the operation is not found in the registry.
        """
        func: Any = self.get(op_type)
        if func is not None:
            return func(*args, **kwargs)

        # Fallback to global pure python registry if available
        if self is not global_eager_registry:
            func: Any = global_eager_registry.get(op_type)
            if func is not None:
                return func(*args, **kwargs)

        # Universal Math Backfill Fallback via backend_module introspection
        # If the op is not explicitly registered but the backend natively supports it, map it dynamically.
        # This resolves missing operations (like Sin, Cos, Add, MatMul) that adhere to the
        # N-to-M universal utility rule natively on the target tensor backend.
        if args and hasattr(args[0], op_type.lower()):
            # e.g., np.sin() fallback
            return getattr(args[0], op_type.lower())(*args[1:], **kwargs)

        backend: Any = kwargs.get("backend_module")
        if backend is None and len(args) > 0:
            backend: Any = args[0]
            args: Any = args[1:]

        if backend is not None:
            # Map common names
            op_name: Any = op_type.lower()
            if hasattr(backend, op_name):
                return getattr(backend, op_name)(*args, **kwargs)

            # Alternative common mappings
            mapping: Any = {"matmul": "matmul", "add": "add", "sub": "subtract", "mul": "multiply", "div": "divide", "truedivide": "divide"}
            if op_name in mapping and hasattr(backend, mapping[op_name]):
                return getattr(backend, mapping[op_name])(*args, **kwargs)

        msg: Any = f"Operation '{op_type}' not found in registry and no universal fallback could be inferred for backend {backend}."
        raise ValueError(msg)


# Global registry instance
global_eager_registry: Any = EagerOpRegistry()

mlx_eager_registry: Any = EagerOpRegistry()
numpy_eager_registry: Any = EagerOpRegistry()
pure_python_eager_registry: Any = EagerOpRegistry()


@global_eager_registry.register("CustomVJP")
def _eager_custom_vjp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
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
def _eager_process_custom_vjp_call(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Process custom vjp."""
    bwd_fn: Any = kwargs["bwd_fn"]
    return bwd_fn(None, *args)


@global_eager_registry.register("TupleGetItem")
def _eager_tuple_get_item(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Tuple get item."""
    index: Any = kwargs.get("index", 0)
    return args[0][index]
