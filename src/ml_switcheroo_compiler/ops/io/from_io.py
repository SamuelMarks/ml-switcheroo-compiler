"""I/O and memory operations."""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import shutil  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerValue
from ml_switcheroo_compiler.core.config import config as core_config  # noqa: F401
from ml_switcheroo_compiler.core.dtype import DType  # noqa: F401
from ml_switcheroo_compiler.core.tensor import Tensor  # noqa: F401
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat  # noqa: F401
from ml_switcheroo_compiler.serialization.utils import load_npz  # noqa: F401


@register_op("Fromfile")
class Fromfile(OpDef):
    """Construct an array from data in a text or binary file.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.
    """

    op_name = "Fromfile"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        count = kwargs.get("count", -1)
        return (count if count != -1 else None,)


@register_op("Fromstring")
class Fromstring(OpDef):
    """Provide a new 1-D array initialized from text data in a string.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.
    """

    op_name = "Fromstring"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        count = kwargs.get("count", -1)
        return (count if count != -1 else None,)


@register_op("Fromiter")
class Fromiter(OpDef):
    """Create a new 1-dimensional array from an iterable Any.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.
    """

    op_name = "Fromiter"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        count = kwargs.get("count", -1)
        return (count if count != -1 else None,)


@register_op("Fromfunction")
class Fromfunction(OpDef):
    """Construct an array by executing a function over each coordinate.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.
    """

    op_name = "Fromfunction"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = kwargs.get("shape", args[1] if len(args) > 1 else ())
        return tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)


def fromfile(file: EagerValue, dtype: EagerValue = float, count: int = -1, sep: str = "", offset: int = 0, *, like: EagerValue = None) -> EagerValue:
    """Construct an array from data in a text or binary file.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.

    Args:
        file (Any): The file parameter.
        like (Any): The like parameter.
    dtype (Any): The dtype parameter.
        count (int): The count parameter.
        sep (str): The sep parameter.
        offset (int): The offset parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromfile", file, dtype=dtype, count=count, sep=sep, offset=offset, like=like)


def fromstring(string: str, dtype: EagerValue = float, count: int = -1, sep: str = "", *, like: EagerValue = None) -> EagerValue:
    """Provide a new 1-D array initialized from text data in a string.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.

    Args:
        string (str): The string parameter.
        like (Any): The like parameter.
    dtype (Any): The dtype parameter.
        count (int): The count parameter.
        sep (str): The sep parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromstring", string, dtype=dtype, count=count, sep=sep, like=like)


def fromiter(iterable: EagerValue, dtype: EagerValue, count: int = -1, *, like: EagerValue = None) -> EagerValue:
    """Create a new 1-dimensional array from an iterable Any.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.

    Args:
        iterable (Any): The iterable parameter.
        like (Any): The like parameter.
    dtype (Any): The dtype parameter.
        count (int): The count parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromiter", iterable, dtype, count=count, like=like)


def fromfunction(function: EagerValue, shape: EagerValue, *, dtype: EagerValue = float, like: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Construct an array by executing a function over each coordinate.

    Args:
        dtype (Any): Dtype.
        like (Any): Like.

    Args:
        function (Any): The function parameter.
        shape (Any): The shape parameter.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Fromfunction", function, shape, dtype=dtype, like=like, **kwargs)
