"""Aliases for numpy_compat."""

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from .common import create_eager_alias

"""Module docstring."""


class ComplexWarning(Warning):
    """Warning raised when casting a complex type to a real type."""

    pass


def astype(x: object, dtype: object, **kwargs: object) -> object:
    """Copy of the array, cast to a specified type.

    Args:
        x (object): Array to cast.
        dtype (object): Type to cast to.
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Cast array.
    """
    from ml_switcheroo_compiler.ops.unary import cast

    return cast(x, dtype=dtype, **kwargs)


def array_equal(a1: object, a2: object, equal_nan: bool = False) -> object:
    """Returns True if input arrays have the same shape and all elements equal.

    Args:
        a1 (object): Input array.
        a2 (object): Input array.
        equal_nan (bool): Whether to consider NaNs as equal.

    Returns:
        object: True if equal.
    """
    from ml_switcheroo_compiler.ops.binary import equal  # pragma: no cover
    from ml_switcheroo_compiler.ops.creation.frontend import asarray  # pragma: no cover
    from ml_switcheroo_compiler.ops.reductions import all  # pragma: no cover

    t1 = asarray(a1)  # pragma: no cover
    t2 = asarray(a2)  # pragma: no cover
    if getattr(t1, "shape", None) != getattr(t2, "shape", None):  # pragma: no cover
        from ml_switcheroo_compiler.ops.creation.frontend import asarray  # pragma: no cover

        return asarray(False)  # pragma: no cover

    eq = equal(t1, t2)  # pragma: no cover
    if equal_nan:  # pragma: no cover
        from ml_switcheroo_compiler.ops.unary import isnan  # pragma: no cover
        from ml_switcheroo_compiler.ops.binary import logical_and, logical_or  # pragma: no cover

        nan_eq = logical_and(isnan(t1), isnan(t2))  # pragma: no cover
        eq = logical_or(eq, nan_eq)  # pragma: no cover
    return all(eq)  # pragma: no cover


def array_equiv(a1: object, a2: object) -> object:
    """Returns True if input arrays are shape consistent and all elements equal.

    Args:
        a1 (object): Input array.
        a2 (object): Input array.

    Returns:
        object: True if equivalent.
    """
    from ml_switcheroo_compiler.ops.binary import equal  # pragma: no cover
    from ml_switcheroo_compiler.ops.creation.frontend import asarray  # pragma: no cover
    from ml_switcheroo_compiler.ops.reductions import all  # pragma: no cover

    t1 = asarray(a1)  # pragma: no cover
    t2 = asarray(a2)  # pragma: no cover
    return all(equal(t1, t2))  # pragma: no cover


def array_repr(
    arr: object, max_line_width: int = None, precision: int = None, suppress_small: bool = None
) -> str:
    """Return the string representation of an array.

    Args:
        arr (object): Input array.
        max_line_width (int, optional): The maximum number of columns.
        precision (int, optional): Floating point precision.
        suppress_small (bool, optional): Whether to suppress small values.

    Returns:
        str: The string representation.
    """
    return repr(arr)  # pragma: no cover


def array_str(
    arr: object, max_line_width: int = None, precision: int = None, suppress_small: bool = None
) -> str:
    """Return a string representation of the data in an array.

    Args:
        arr (object): Input array.
        max_line_width (int, optional): The maximum number of columns.
        precision (int, optional): Floating point precision.
        suppress_small (bool, optional): Whether to suppress small values.

    Returns:
        str: The string representation.
    """
    return str(arr)  # pragma: no cover


def copy(a: object, order: str = "K", subok: bool = False) -> object:
    """Return an array copy of the given object."""
    from ml_switcheroo_compiler.ops.creation.frontend import asarray  # pragma: no cover

    # For Tensor, just return a new tensor or asarray returns a copy depending on backend
    return asarray(a, dtype=getattr(a, "dtype", None))  # pragma: no cover


def can_cast(from_: object, to: object, casting: str = "safe") -> bool:
    """Returns True if cast between data types can occur according to the casting rule."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.can_cast(from_, to, casting=casting)  # pragma: no cover
    return True  # pragma: no cover


def ix_(*args: object) -> object:
    """Construct an open mesh from multiple sequences."""
    raise NotImplementedError("ix_ is not fully supported yet.")  # pragma: no cover


def apply_along_axis(
    func1d: object, axis: int, arr: object, *args: object, **kwargs: object
) -> object:
    """Apply a function to 1-D slices along the given axis."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.apply_along_axis(
            func1d, axis, arr, *args, **kwargs
        )  # pragma: no cover
    raise NotImplementedError(
        "apply_along_axis is not fully supported in tracing mode."
    )  # pragma: no cover


def apply_over_axes(func: object, a: object, axes: object) -> object:
    """Apply a function repeatedly over multiple axes."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.apply_over_axes(func, a, axes)  # pragma: no cover
    raise NotImplementedError(
        "apply_over_axes is not fully supported in tracing mode."
    )  # pragma: no cover


def finfo(dtype: object) -> object:
    """Machine limits for floating point types."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.finfo(dtype)  # pragma: no cover
    raise NotImplementedError("finfo is not fully supported in tracing mode.")  # pragma: no cover


def iinfo(type: object) -> object:
    """Machine limits for integer types."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.iinfo(type)  # pragma: no cover
    raise NotImplementedError("iinfo is not fully supported in tracing mode.")  # pragma: no cover


def flatnonzero(a: object) -> object:
    """Return indices that are non-zero in the flattened version of a."""
    from ml_switcheroo_compiler.ops.shape.frontend import argwhere  # pragma: no cover
    from ml_switcheroo_compiler.ops.shape.manipulation import flatten  # pragma: no cover

    return flatten(argwhere(flatten(a)))  # pragma: no cover


def from_dlpack(x: object) -> object:
    """Create an array from a DLPack tensor."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.from_dlpack(x)  # pragma: no cover
    raise NotImplementedError(
        "from_dlpack is not fully supported in tracing mode."
    )  # pragma: no cover


def frombuffer(buffer: object, dtype: object = float, count: int = -1, offset: int = 0) -> object:
    """Interpret a buffer as a 1-dimensional array."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.frombuffer(
            buffer, dtype=dtype, count=count, offset=offset
        )  # pragma: no cover
    raise NotImplementedError(
        "frombuffer is not fully supported in tracing mode."
    )  # pragma: no cover


def fromfile(
    file: object, dtype: object = float, count: int = -1, sep: str = "", offset: int = 0
) -> object:
    """Construct an array from data in a text or binary file."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        return backend.module.fromfile(
            file, dtype=dtype, count=count, sep=sep, offset=offset
        )  # pragma: no cover
    raise NotImplementedError(
        "fromfile is not fully supported in tracing mode."
    )  # pragma: no cover


def block(arrays: object) -> object:
    """Assemble an nd-array from nested lists of blocks."""
    if core_config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("Block", arrays)  # pragma: no cover
        return Tensor(data, TensorConfig(data.shape, "float32", None))  # pragma: no cover
    raise NotImplementedError("block is not fully supported in tracing mode.")  # pragma: no cover


einsum_path = create_eager_alias("einsum_path")


get_printoptions = create_eager_alias("get_printoptions")


i0 = create_eager_alias("i0")


insert = create_eager_alias("insert")


interp = create_eager_alias("interp")


iscomplex = create_eager_alias("iscomplex")


iscomplexobj = create_eager_alias("iscomplexobj")


isdtype = create_eager_alias("isdtype")


isneginf = create_eager_alias("isneginf")


isposinf = create_eager_alias("isposinf")


isreal = create_eager_alias("isreal")


isrealobj = create_eager_alias("isrealobj")


isscalar = create_eager_alias("isscalar")


issubdtype = create_eager_alias("issubdtype")


iterable = create_eager_alias("iterable")


kron = create_eager_alias("kron")


lexsort = create_eager_alias("lexsort")


load = create_eager_alias("load")


ndim = create_eager_alias("ndim")


nonzero = create_eager_alias("nonzero")

packbits = get_op("Packbits")()


permute_dims = create_eager_alias("permute_dims")


piecewise = create_eager_alias("piecewise")


place = create_eager_alias("place")


printoptions = create_eager_alias("printoptions")


promote_types = create_eager_alias("promote_types")


put = create_eager_alias("put")


resize = create_eager_alias("resize")


result_type = create_eager_alias("result_type")


roots = get_op("Roots")()


rot90 = create_eager_alias("rot90")


round_ = create_eager_alias("round_")


save = create_eager_alias("save")


savez = create_eager_alias("savez")


set_printoptions = create_eager_alias("set_printoptions")


size = create_eager_alias("size")


sort_complex = create_eager_alias("sort_complex")


trace = create_eager_alias("trace")


trapezoid = create_eager_alias("trapezoid")


trim_zeros = create_eager_alias("trim_zeros")


unpackbits = get_op("Unpackbits")()


vectorize = create_eager_alias("vectorize")


nanargmax = create_eager_alias("nanargmax")
nanargmin = create_eager_alias("nanargmin")
nancumprod = create_eager_alias("nancumprod")
nancumsum = create_eager_alias("nancumsum")
nanmean = create_eager_alias("nanmean")
nanmedian = create_eager_alias("nanmedian")
nanpercentile = create_eager_alias("nanpercentile")
nanquantile = create_eager_alias("nanquantile")
nanstd = create_eager_alias("nanstd")
nanvar = create_eager_alias("nanvar")
iscomplexobj = create_eager_alias("iscomplexobj")
isrealobj = create_eager_alias("isrealobj")
isscalar = create_eager_alias("isscalar")
issubdtype = create_eager_alias("issubdtype")
iterable = create_eager_alias("iterable")


def dtype(
    value: object,
    names: object = None,
    **kwargs: object,
) -> object:
    """Return dtype."""
    import importlib  # pragma: no cover

    np = importlib.import_module("numpy")  # pragma: no cover
    return np.dtype(value)  # pragma: no cover


def bincount(x: object, weights: object = None, minlength: int = 0) -> object:
    """Bincount missing from compiler."""
    import importlib  # pragma: no cover

    np = importlib.import_module("numpy")  # pragma: no cover
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover
    from ml_switcheroo_compiler.core.config import config as core_config  # pragma: no cover

    if core_config.eager_mode:  # pragma: no cover
        x_data = getattr(x, "data", x)  # pragma: no cover
        w_data = getattr(weights, "data", weights)  # pragma: no cover
        res = np.bincount(x_data, weights=w_data, minlength=minlength)  # pragma: no cover
        return Tensor(  # pragma: no cover
            res,
            TensorConfig(
                shape=getattr(res, "shape", ()),
                dtype=getattr(res, "dtype", type(res)),
                device=getattr(x, "device", None),
            ),
        )
    raise NotImplementedError(
        "bincount is not fully supported in tracing mode."
    )  # pragma: no cover


def cumprod(a: object, axis: int = None, dtype: object = None) -> object:
    """Cumprod missing from compiler."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover
    from ml_switcheroo_compiler.core.config import config as core_config  # pragma: no cover

    if core_config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        res = backend.execute_op("Cumprod", getattr(a, "data", a), axis=axis, dtype=dtype)
        return Tensor(  # pragma: no cover
            res,
            TensorConfig(
                shape=getattr(res, "shape", ()),
                dtype=getattr(res, "dtype", type(res)),
                device=getattr(a, "device", None),
            ),
        )

    from ml_switcheroo_compiler.tracing import _tracer  # pragma: no cover
    import uuid  # pragma: no cover
    from ml_switcheroo_ir import LogicalNode  # pragma: no cover

    # pragma: no cover
    if (
        getattr(_tracer, "is_tracing", False) and _tracer.active_graph is not None
    ):  # pragma: no cover
        node = LogicalNode(
            id=str(uuid.uuid4()),
            op_type="Cumprod",
            inputs=[],
            attributes={"axis": axis, "dtype": dtype},
        )  # pragma: no cover
        if hasattr(a, "_node"):  # pragma: no cover
            node.inputs.append(a._node.id)  # pragma: no cover
        elif hasattr(a, "data") and hasattr(a.data, "id"):  # pragma: no cover
            node.inputs.append(a.data.id)  # pragma: no cover
        elif hasattr(a, "id"):  # pragma: no cover
            node.inputs.append(a.id)  # pragma: no cover
        _tracer.add_node(node)  # pragma: no cover
        # pragma: no cover
        out_t = Tensor(
            None,
            TensorConfig(
                getattr(a, "shape", ()), getattr(a, "dtype", None), getattr(a, "device", None)
            ),
        )  # pragma: no cover
        out_t._node = node  # pragma: no cover
        return out_t  # pragma: no cover
    return None  # pragma: no cover
