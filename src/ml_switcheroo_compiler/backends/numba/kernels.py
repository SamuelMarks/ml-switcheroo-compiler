"""Native Numba array and mathematical kernels for CPU execution."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter, adapt_backend_module

float32: str = "float32"


def _resolve_dense_backend() -> EagerBackendAdapter:
    """Resolve the underlying array buffer backend via the active backend registry.

    Returns:
        EagerBackendAdapter: Registered host array backend adapter.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend

    active = get_active_backend()
    if active.__name__ in ("NumbaGenerator", "NumbaCodeGenerator"):
        return adapt_backend_module(BackendRegistry.get("pure_python"), backend_name="pure_python")
    return adapt_backend_module(active.get_module())


def _make_nb_kernel(op_name: str) -> BackendArg:
    """Create a lazy Numba kernel wrapper delegating to the dense backend.

    Args:
        op_name (str): Target operation name.

    Returns:
        BackendArg: Lazy wrapper callable.
    """

    def _kernel(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute kernel on dense backend.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Computed kernel result.
        """
        fn = _resolve_dense_backend().require_op(op_name)
        return fn(*args, **kwargs)

    return _kernel


abs = _make_nb_kernel("abs")
all = _make_nb_kernel("all")
any = _make_nb_kernel("any")
arange = _make_nb_kernel("arange")
arccos = _make_nb_kernel("arccos")
arccosh = _make_nb_kernel("arccosh")
arcsin = _make_nb_kernel("arcsin")
arcsinh = _make_nb_kernel("arcsinh")
arctan = _make_nb_kernel("arctan")
arctan2 = _make_nb_kernel("arctan2")
arctanh = _make_nb_kernel("arctanh")
argmax = _make_nb_kernel("argmax")
argmin = _make_nb_kernel("argmin")
argsort = _make_nb_kernel("argsort")
bitwise_and = _make_nb_kernel("bitwise_and")
bitwise_or = _make_nb_kernel("bitwise_or")
bitwise_xor = _make_nb_kernel("bitwise_xor")
ceil = _make_nb_kernel("ceil")
clip = _make_nb_kernel("clip")
concatenate = _make_nb_kernel("concatenate")
cos = _make_nb_kernel("cos")
cosh = _make_nb_kernel("cosh")
cumprod = _make_nb_kernel("cumprod")
cumsum = _make_nb_kernel("cumsum")
diag = _make_nb_kernel("diag")
divide = _make_nb_kernel("divide")
dot = _make_nb_kernel("dot")
equal = _make_nb_kernel("equal")
exp = _make_nb_kernel("exp")
exp2 = _make_nb_kernel("exp2")
expand_dims = _make_nb_kernel("expand_dims")
expm1 = _make_nb_kernel("expm1")
eye = _make_nb_kernel("eye")
floor = _make_nb_kernel("floor")
full = _make_nb_kernel("full")
greater = _make_nb_kernel("greater")
hypot = _make_nb_kernel("hypot")
identity = _make_nb_kernel("identity")
invert = _make_nb_kernel("invert")
left_shift = _make_nb_kernel("left_shift")
less = _make_nb_kernel("less")
linspace = _make_nb_kernel("linspace")
log = _make_nb_kernel("log")
log10 = _make_nb_kernel("log10")
log1p = _make_nb_kernel("log1p")
log2 = _make_nb_kernel("log2")
logical_and = _make_nb_kernel("logical_and")
logical_not = _make_nb_kernel("logical_not")
logical_or = _make_nb_kernel("logical_or")
logical_xor = _make_nb_kernel("logical_xor")
matmul = _make_nb_kernel("matmul")
max = _make_nb_kernel("max")
maximum = _make_nb_kernel("maximum")
mean = _make_nb_kernel("mean")
min = _make_nb_kernel("min")
minimum = _make_nb_kernel("minimum")
multiply = _make_nb_kernel("multiply")
negative = _make_nb_kernel("negative")
not_equal = _make_nb_kernel("not_equal")
ones = _make_nb_kernel("ones")
power = _make_nb_kernel("power")
prod = _make_nb_kernel("prod")
ravel = _make_nb_kernel("ravel")
reciprocal = _make_nb_kernel("reciprocal")
reshape = _make_nb_kernel("reshape")
right_shift = _make_nb_kernel("right_shift")
rint = _make_nb_kernel("rint")
sign = _make_nb_kernel("sign")
sin = _make_nb_kernel("sin")
sinh = _make_nb_kernel("sinh")
sort = _make_nb_kernel("sort")
sqrt = _make_nb_kernel("sqrt")
square = _make_nb_kernel("square")
squeeze = _make_nb_kernel("squeeze")
stack = _make_nb_kernel("stack")
std = _make_nb_kernel("std")
subtract = _make_nb_kernel("subtract")
sum = _make_nb_kernel("sum")
tan = _make_nb_kernel("tan")
tanh = _make_nb_kernel("tanh")
transpose = _make_nb_kernel("transpose")
tril = _make_nb_kernel("tril")
triu = _make_nb_kernel("triu")
true_divide = _make_nb_kernel("true_divide")
trunc = _make_nb_kernel("trunc")
var = _make_nb_kernel("var")
where = _make_nb_kernel("where")


def asarray(data: BackendArg, dtype: BackendArg = None) -> BackendArg:
    """Convert input sequence or buffer into a Numba-compatible array buffer.

    Args:
        data (BackendArg): Input sequence, scalar, or array buffer.
        dtype (BackendArg): Optional target datatype specifier.

    Returns:
        BackendArg: Contiguous array buffer compatible with Numba JIT kernels.
    """
    backend = _resolve_dense_backend()
    if dtype is not None:
        return backend.require_op("asarray")(data, dtype=dtype)
    return backend.asarray(data)


def array(data: BackendArg, dtype: BackendArg = None) -> BackendArg:
    """Construct a new Numba-compatible array from input data.

    Args:
        data (BackendArg): Input sequence or array buffer.
        dtype (BackendArg): Optional target datatype specifier.

    Returns:
        BackendArg: Newly allocated array buffer.
    """
    backend = _resolve_dense_backend()
    if dtype is not None:
        return backend.require_op("array")(data, dtype=dtype)
    return backend.require_op("array")(data)


def zeros(shape: tuple[int, ...] | int, dtype: BackendArg = "float32") -> BackendArg:
    """Allocate a zero-initialized array buffer of the given shape.

    Args:
        shape (tuple[int, ...] | int): Target shape dimensions.
        dtype (BackendArg): Target datatype. Defaults to 'float32'.

    Returns:
        BackendArg: Zero-initialized array buffer.
    """
    backend = _resolve_dense_backend()
    return backend.require_op("zeros")(shape, dtype=dtype)


def zeros_like(arr: BackendArg) -> BackendArg:
    """Allocate a zero-initialized array matching the shape and dtype of arr.

    Args:
        arr (BackendArg): Reference array buffer.

    Returns:
        BackendArg: Zero-initialized array buffer.
    """
    backend = _resolve_dense_backend()
    return backend.require_op("zeros_like")(arr)


def add(a: BackendArg, b: BackendArg) -> BackendArg:
    """Compute elementwise addition of two operands.

    Args:
        a (BackendArg): First operand.
        b (BackendArg): Second operand.

    Returns:
        BackendArg: Elementwise sum.
    """
    backend = _resolve_dense_backend()
    return backend.require_op("add")(a, b)
