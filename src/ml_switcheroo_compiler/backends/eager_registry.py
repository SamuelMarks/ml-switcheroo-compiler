"""eager_registry.py module."""

import builtins
from typing import Callable, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class EagerTensorProtocol(Protocol):
    """Protocol defining the structural type contract for eager backend array/tensor objects."""

    def __getitem__(self, key: Union[int, slice, tuple, list, str, "builtins.ellipsis", None]) -> "EagerTensorProtocol":
        """Get an item or slice from the array."""
        ...

    def __setitem__(self, key: Union[int, slice, tuple, list, str, "builtins.ellipsis", None], value: "EagerTensorProtocol") -> None:
        """Set an item or slice in the array."""
        ...

    def __len__(self) -> int:
        """Return the length of the leading dimension."""
        ...

    def __neg__(self) -> "EagerTensorProtocol":
        """Elementwise negation."""
        ...

    def __add__(self, other: "EagerValue") -> "EagerTensorProtocol":
        """Add two arrays."""
        ...

    def __sub__(self, other: "EagerValue") -> "EagerTensorProtocol":
        """Subtract two arrays."""
        ...

    def __mul__(self, other: "EagerValue") -> "EagerTensorProtocol":
        """Multiply two arrays."""
        ...

    def __truediv__(self, other: "EagerValue") -> "EagerTensorProtocol":
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

    def __pow__(self, other: "EagerValue") -> "EagerTensorProtocol":
        """Power operation."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        ...

    @property
    def dtype(self) -> object:
        """Get the dtype of the array."""
        ...


BackendArray = EagerTensorProtocol
EagerValue = Union[int, float, list, tuple, str, bool, EagerTensorProtocol, None]

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
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("CustomJVP")
def _eager_custom_jvp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Custom jvp."""
    fun = kwargs.get("fun")
    if callable(fun):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            return fun(*args)
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("ProcessCustomVJPCall")
def _eager_process_custom_vjp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Process custom vjp."""
    bwd_fn = kwargs["bwd_fn"]
    if callable(bwd_fn):
        return bwd_fn.__call__(None, *args)
    return None


@global_eager_registry.register("ProcessCustomJVPCall")
def _eager_process_custom_jvp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Process custom jvp."""
    jvp_rule = kwargs.get("jvp_rule")
    num_primals = kwargs.get("num_primals", len(args) // 2)
    primals = args[:num_primals]
    tangents = args[num_primals:]
    if callable(jvp_rule):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            try:
                res = jvp_rule(primals, tangents)
            except TypeError:
                res = jvp_rule(*primals, *tangents)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            return res
        return (None, res)
    return (None, None)


@global_eager_registry.register("TupleGetItem")
def _eager_tuple_get_item(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Tuple get item."""
    index: int = int(getattr(kwargs.get("index", 0), "__int__", lambda: 0)())
    if isinstance(args[0], (tuple, list)):
        return args[0][index]
    return None
