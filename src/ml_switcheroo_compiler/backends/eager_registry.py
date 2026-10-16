"""eager_registry.py module."""

from __future__ import annotations

import builtins
import ctypes
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Optional, Protocol, Union, cast, runtime_checkable

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.sparse.types import COOTensor
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.utils import GradTensorProtocol
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalGraph, LogicalNode
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode, SymVar


@runtime_checkable
class DataclassProtocol(Protocol):
    """Protocol matching any dataclass configuration or metadata object."""

    @property
    def __dataclass_fields__(self) -> Iterable[str]:
        """Return dataclass fields."""
        ...


@runtime_checkable
class ShapeProtocol(Protocol):
    """Protocol for any array-like object exposing a shape tuple."""

    @property
    def shape(self) -> Sequence[Union[int, str]]:
        """Get the shape of the array.

        Returns:
            Sequence[Union[int, str]]: Array shape sequence.
        """
        ...


EagerOperand = Union[
    int,
    float,
    complex,
    str,
    bool,
    slice,
    type,
    ModuleType,
    "builtins.ellipsis",
    "EagerTensorProtocol",
    ShapeProtocol,
    Sequence[Union[int, float, str, slice, "builtins.ellipsis", "EagerTensorProtocol", None]],
    None,
]


@runtime_checkable
class EagerTensorProtocol(Protocol):
    """Protocol defining the structural type contract for eager backend array/tensor objects."""

    if TYPE_CHECKING:
        size: int
        ndim: int
        T: EagerTensorProtocol
        real: EagerTensorProtocol
        imag: EagerTensorProtocol
        flat: EagerTensorProtocol
        copy: Callable[[], EagerTensorProtocol]
        astype: Callable[..., EagerTensorProtocol]
        reshape: Callable[..., EagerTensorProtocol]
        transpose: Callable[..., EagerTensorProtocol]
        swapaxes: Callable[..., EagerTensorProtocol]
        flatten: Callable[[], EagerTensorProtocol]
        ravel: Callable[[], EagerTensorProtocol]
        tolist: Callable[[], EagerOperand]
        tobytes: Callable[[], bytes]
        item: Callable[..., float | int | bool | complex]
        sum: Callable[..., EagerTensorProtocol]
        mean: Callable[..., EagerTensorProtocol]
        max: Callable[..., EagerTensorProtocol]
        min: Callable[..., EagerTensorProtocol]
        prod: Callable[..., EagerTensorProtocol]
        std: Callable[..., EagerTensorProtocol]
        var: Callable[..., EagerTensorProtocol]
        conj: Callable[..., EagerTensorProtocol]
        conjugate: Callable[..., EagerTensorProtocol]
        squeeze: Callable[..., EagerTensorProtocol]
        clip: Callable[..., EagerTensorProtocol]
        dot: Callable[..., EagerTensorProtocol]
        argsort: Callable[..., EagerTensorProtocol]

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerTensorProtocol:
            """Call operator when protocol represents a callable backend function.

            Args:
                *args (EagerValue): Positional arguments.
                **kwargs (EagerValue): Keyword arguments.

            Returns:
                EagerTensorProtocol: Result.
            """
            ...

        def __getattr__(self, name: str) -> EagerTensorProtocol:
            """Dynamic attribute access.

            Args:
                name (str): Attribute name.

            Returns:
                EagerTensorProtocol: Attribute value.
            """
            ...

        def __eq__(self, other: object) -> bool:
            """Equality comparison.

            Args:
                other (object): Other value.

            Returns:
                bool: Equality result.
            """
            ...

        def __contains__(self, item: EagerValue) -> bool:
            """Membership check.

            Args:
                item (EagerValue): Item to check.

            Returns:
                bool: True if item is contained.
            """
            ...

        def __ne__(self, other: object) -> bool:
            """Inequality comparison.

            Args:
                other (object): Other value.

            Returns:
                bool: Inequality result.
            """
            ...

        def __enter__(self) -> EagerTensorProtocol:
            """Enter context manager.

            Returns:
                EagerTensorProtocol: Context object.
            """
            ...

        def __await__(self) -> Generator[None, None, EagerTensorProtocol]:
            """Await coroutine tensor/stream operation.

            Returns:
                Generator[None, None, EagerTensorProtocol]: Awaitable generator.
            """
            ...

        def __exit__(self, exc_type: EagerValue, exc_val: EagerValue, exc_tb: EagerValue) -> EagerValue:
            """Exit context manager.

            Args:
                exc_type (EagerValue): Exception type.
                exc_val (EagerValue): Exception value.
                exc_tb (EagerValue): Traceback.

            Returns:
                EagerValue: Suppress flag.
            """
            ...

        def __iter__(self) -> builtins.map[EagerTensorProtocol]:
            """Iterate over the leading dimension.

            Returns:
                builtins.map[EagerTensorProtocol]: Iterator over leading dimension slices.
            """
            ...

        def __int__(self) -> int:
            """Convert scalar tensor to int.

            Returns:
                int: Integer value.
            """
            ...

        def __index__(self) -> int:
            """Convert scalar tensor to index int.

            Returns:
                int: Index integer value.
            """
            ...

        def __fspath__(self) -> str:
            """Return file system path representation.

            Returns:
                str: File system path string.
            """
            ...

        def __pos__(self) -> EagerTensorProtocol:
            """Unary positive.

            Returns:
                EagerTensorProtocol: Positive array.
            """
            ...

        def __lshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Left bitwise shift.

            Args:
                other (EagerValue): Shift amount.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __rshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Right bitwise shift.

            Args:
                other (EagerValue): Shift amount.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __rlshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse left bitwise shift.

            Args:
                other (EagerValue): Left operand.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __rrshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse right bitwise shift.

            Args:
                other (EagerValue): Left operand.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __float__(self) -> float:
            """Convert scalar tensor to float.

            Returns:
                float: Float value.
            """
            ...

        def __abs__(self) -> EagerTensorProtocol:
            """Elementwise absolute value.

            Returns:
                EagerTensorProtocol: Absolute value array.
            """
            ...

        def __invert__(self) -> EagerTensorProtocol:
            """Elementwise bitwise inversion.

            Returns:
                EagerTensorProtocol: Inverted array.
            """
            ...

        def __and__(self, other: EagerValue) -> EagerTensorProtocol:
            """Bitwise AND.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Result array.
            """
            ...

        def __or__(self, other: EagerValue) -> EagerTensorProtocol:
            """Bitwise OR.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Result array.
            """
            ...

        def __xor__(self, other: EagerValue) -> EagerTensorProtocol:
            """Bitwise XOR.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Result array.
            """
            ...

        def __radd__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse add.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Sum array.
            """
            ...

        def __rsub__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse subtract.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Difference array.
            """
            ...

        def __rmul__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse multiply.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Product array.
            """
            ...

        def __rtruediv__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse true divide.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Quotient array.
            """
            ...

        def __floordiv__(self, other: EagerValue) -> EagerTensorProtocol:
            """Floor divide.

            Args:
                other (EagerValue): Right-hand divisor.

            Returns:
                EagerTensorProtocol: Floor quotient array.
            """
            ...

        def __rfloordiv__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse floor divide.

            Args:
                other (EagerValue): Left-hand dividend.

            Returns:
                EagerTensorProtocol: Floor quotient array.
            """
            ...

        def __mod__(self, other: EagerValue) -> EagerTensorProtocol:
            """Modulo operation.

            Args:
                other (EagerValue): Right-hand divisor.

            Returns:
                EagerTensorProtocol: Remainder array.
            """
            ...

        def __rmod__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse modulo operation.

            Args:
                other (EagerValue): Left-hand dividend.

            Returns:
                EagerTensorProtocol: Remainder array.
            """
            ...

        def __matmul__(self, other: EagerValue) -> EagerTensorProtocol:
            """Matrix multiplication.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Matrix product array.
            """
            ...

        def __rmatmul__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse matrix multiplication.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Matrix product array.
            """
            ...

        def __rpow__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse power operation.

            Args:
                other (EagerValue): Base operand.

            Returns:
                EagerTensorProtocol: Exponentiated array.
            """
            ...

    def __getitem__(
        self,
        key: Union[
            int,
            slice,
            tuple[EagerOperand, ...],
            list[EagerOperand],
            str,
            builtins.ellipsis,
            EagerTensorProtocol,
            EagerValue,
            None,
        ],
    ) -> EagerTensorProtocol:
        """Get an item or slice from the array.

        Args:
            key (Union[int, slice, tuple, list, str, builtins.ellipsis, EagerTensorProtocol, None]): Indexing key or slice.

        Returns:
            EagerTensorProtocol: Indexed tensor view or copy.
        """
        ...

    def __setitem__(
        self,
        key: Union[
            int,
            slice,
            tuple[EagerOperand, ...],
            list[EagerOperand],
            str,
            builtins.ellipsis,
            EagerTensorProtocol,
            EagerValue,
            None,
        ],
        value: EagerValue,
    ) -> None:
        """Set an item or slice in the array.

        Args:
            key (Union[int, slice, tuple, list, str, builtins.ellipsis, EagerTensorProtocol, None]): Target index or slice.
            value (EagerValue): Value to assign.
        """
        ...

    def __len__(self) -> int:
        """Return the length of the leading dimension.

        Returns:
            int: Leading dimension size.
        """
        ...

    def __neg__(self) -> EagerTensorProtocol:
        """Elementwise negation.

        Returns:
            EagerTensorProtocol: Negated array.
        """
        ...

    def __add__(self, other: EagerValue) -> EagerTensorProtocol:
        """Add two arrays.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Sum array.
        """
        ...

    def __sub__(self, other: EagerValue) -> EagerTensorProtocol:
        """Subtract two arrays.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Difference array.
        """
        ...

    def __mul__(self, other: EagerValue) -> EagerTensorProtocol:
        """Multiply two arrays.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Product array.
        """
        ...

    def __truediv__(self, other: EagerValue) -> EagerTensorProtocol:
        """Divide two arrays.

        Args:
            other (EagerValue): Right-hand divisor.

        Returns:
            EagerTensorProtocol: Quotient array.
        """
        ...

    def __lt__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare less than.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __le__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare less than or equal.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __gt__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare greater than.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __ge__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare greater than or equal.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __pow__(self, other: EagerValue) -> EagerTensorProtocol:
        """Power operation.

        Args:
            other (EagerValue): Exponent operand.

        Returns:
            EagerTensorProtocol: Exponentiated array.
        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array.

        Returns:
            tuple[int, ...]: Array shape tuple.
        """
        ...

    @property
    def dtype(self) -> str | type | None:
        """Get the dtype of the array.

        Returns:
            str | type | None: Array data type.
        """
        ...


BackendArray = EagerTensorProtocol
EagerValue = Union[
    int,
    float,
    complex,
    Sequence["EagerValue"],
    Mapping["EagerValue", "EagerValue"],
    Mapping[str, "EagerValue"],
    Mapping[Union[str, int], "EagerValue"],
    set["EagerValue"],
    slice,
    str,
    bool,
    Callable[..., "EagerValue"],
    type,
    ModuleType,
    "builtins.ellipsis",
    "ctypes.Array[ctypes.c_char]",
    "ctypes.Array[ctypes.c_void_p]",
    "COOTensor",
    "TensorLike",
    "GradTensorProtocol",
    "Tensor",
    "SymVar",
    "SymNode",
    "SymInt",
    "LogicalNode",
    "LogicalGraph",
    "IRGraph",
    DataclassProtocol,
    EagerTensorProtocol,
    ShapeProtocol,
    None,
]


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
            return cast(EagerValue, getattr(args[0], op_type.lower())(*args[1:], **kwargs))

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
                return cast(EagerValue, getattr(backend, op_name)(*op_args, **kwargs))

            # Alternative common mappings
            mapping = {"matmul": "matmul", "add": "add", "sub": "subtract", "mul": "multiply", "div": "divide", "truedivide": "divide"}
            if op_name in mapping and hasattr(backend, mapping[op_name]):
                return cast(EagerValue, getattr(backend, mapping[op_name])(*op_args, **kwargs))

        msg = f"Operation '{op_type}' not found in registry and no universal fallback could be inferred for backend {backend}."
        raise ValueError(msg)


# Global registry instance
global_eager_registry = EagerOpRegistry()

mlx_eager_registry = EagerOpRegistry()
numpy_eager_registry = EagerOpRegistry()
pure_python_eager_registry = EagerOpRegistry()


@global_eager_registry.register("CustomVJP")
def _eager_custom_vjp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate forward pass of a CustomVJP operation in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Primal input arguments.
        **kwargs (EagerValue): Keyword attributes.

    Returns:
        EagerValue: Forward output value or tuple.
    """
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("CustomJVP")
def _eager_custom_jvp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate forward pass of a CustomJVP operation in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Primal input arguments.
        **kwargs (EagerValue): Keyword attributes containing optional 'fun'.

    Returns:
        EagerValue: Forward output value or tuple.
    """
    fun = kwargs.get("fun")
    if callable(fun):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            return fun(*args)
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("ProcessCustomVJPCall")
def _eager_process_custom_vjp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute custom backward VJP callback in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Cotangent arguments.
        **kwargs (EagerValue): Keyword attributes containing 'bwd_fn'.

    Returns:
        EagerValue: Computed backward gradients.
    """
    bwd_fn = kwargs["bwd_fn"]
    if callable(bwd_fn):
        return cast(Callable[..., EagerValue], bwd_fn)(None, *args)
    return None


@global_eager_registry.register("ProcessCustomJVPCall")
def _eager_process_custom_jvp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute custom forward JVP rule callback in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Concatenated primal and tangent arguments.
        **kwargs (EagerValue): Keyword attributes containing 'jvp_rule' and 'num_primals'.

    Returns:
        EagerValue: Tuple of (primal_out, tangent_out).
    """
    jvp_rule = kwargs.get("jvp_rule")
    num_primals = kwargs.get("num_primals", len(args) // 2)
    primals = args[: cast("int | None", num_primals)]
    tangents = args[cast("int | None", num_primals) :]
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
    """Extract an element from a tuple or list by index in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Input sequence argument.
        **kwargs (EagerValue): Keyword attributes containing 'index'.

    Returns:
        EagerValue: Extracted element at the requested index.
    """
    index: int = int(getattr(kwargs.get("index", 0), "__int__", lambda: 0)())
    if isinstance(args[0], (tuple, list)):
        return cast(EagerValue, args[0][index])
    return None
