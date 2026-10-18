"""Array construction operations from files, strings, iterables, and coordinate functions."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Fromfile")
class Fromfile(OpDef):
    """Construct an array from data in a text or binary file."""

    op_name = "Fromfile"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D output shape for Fromfile based on `count`.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword attributes including `count`.

        Returns:
            TensorOrScalar: Inferred 1-tuple shape.
        """
        _ = args
        count = kwargs.get("count", -1)
        return (count if count != -1 else None,)


@register_op("Fromstring")
class Fromstring(OpDef):
    """Construct a 1D array initialized from text data in a string."""

    op_name = "Fromstring"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D output shape for Fromstring based on `count`.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword attributes including `count`.

        Returns:
            TensorOrScalar: Inferred 1-tuple shape.
        """
        _ = args
        count = kwargs.get("count", -1)
        return (count if count != -1 else None,)


@register_op("Fromiter")
class Fromiter(OpDef):
    """Create a 1D array from an iterable object."""

    op_name = "Fromiter"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D output shape for Fromiter based on `count`.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword attributes including `count`.

        Returns:
            TensorOrScalar: Inferred 1-tuple shape.
        """
        _ = args
        count = kwargs.get("count", -1)
        return (count if count != -1 else None,)


@register_op("Fromfunction")
class Fromfunction(OpDef):
    """Construct an array by executing a function over each coordinate."""

    op_name = "Fromfunction"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for Fromfunction from `shape` argument.

        Args:
            *args (TensorOrScalar): Positional arguments (function, shape).
            **kwargs (TensorOrScalar): Keyword attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        shape = kwargs.get("shape", args[1] if len(args) > 1 else ())
        return tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)


def fromfile(
    file: TensorOrScalar,
    dtype: TensorOrScalar = float,
    count: int = -1,
    sep: str = "",
    offset: int = 0,
    *,
    like: TensorOrScalar = None,
) -> TensorOrScalar:
    """Construct an array from data in a text or binary file.

    Args:
        file (TensorOrScalar): Open file object or filename path.
        dtype (TensorOrScalar): Data type of the returned array.
        count (int): Number of items to read (-1 means all items).
        sep (str): Separator between items if file is a text file.
        offset (int): Byte offset from the file's current position.
        like (TensorOrScalar): Reference array prototype.

    Returns:
        TensorOrScalar: Array constructed from the file contents.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromfile", file, dtype=dtype, count=count, sep=sep, offset=offset, like=like)


def fromstring(
    string: str,
    dtype: TensorOrScalar = float,
    count: int = -1,
    sep: str = "",
    *,
    like: TensorOrScalar = None,
) -> TensorOrScalar:
    """Construct a 1-D array initialized from text data in a string.

    Args:
        string (str): Input string containing the data.
        dtype (TensorOrScalar): Data type of the resulting array.
        count (int): Number of items to read (-1 means all data).
        sep (str): String separating numbers in the input data.
        like (TensorOrScalar): Reference array prototype.

    Returns:
        TensorOrScalar: 1-D tensor constructed from the parsed string.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromstring", string, dtype=dtype, count=count, sep=sep, like=like)


def fromiter(
    iterable: TensorOrScalar,
    dtype: TensorOrScalar,
    count: int = -1,
    *,
    like: TensorOrScalar = None,
) -> TensorOrScalar:
    """Create a 1-dimensional array from an iterable object.

    Args:
        iterable (TensorOrScalar): Iterable object providing data for the array.
        dtype (TensorOrScalar): Data type of the returned array.
        count (int): Number of items to read from `iterable` (-1 means all data).
        like (TensorOrScalar): Reference array prototype.

    Returns:
        TensorOrScalar: 1-D tensor populated from `iterable`.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromiter", iterable, dtype, count=count, like=like)


def fromfunction(
    function: TensorOrScalar,
    shape: TensorOrScalar,
    *,
    dtype: TensorOrScalar = float,
    like: TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """Construct an array by executing a function over each coordinate.

    Args:
        function (TensorOrScalar): Callable invoked with coordinate arrays.
        shape (TensorOrScalar): Output array shape tuple.
        dtype (TensorOrScalar): Data type of the coordinate grid passed to `function`.
        like (TensorOrScalar): Reference array prototype.
        **kwargs (TensorOrScalar): Additional keyword arguments passed to `function`.

    Returns:
        TensorOrScalar: Tensor result of evaluating `function` on the coordinate grid.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromfunction", function, shape, dtype=dtype, like=like, **kwargs)
