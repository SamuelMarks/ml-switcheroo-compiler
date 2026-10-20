"""Filesystem and raw binary file I/O operations."""

from __future__ import annotations

import glob
import os
import shutil

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def read_file(filename: str | Tensor, name: TensorOrScalar = None) -> TensorOrScalar:
    """Read the entire contents of a file into a byte tensor.

    Args:
        filename (str | Tensor): Path to the file to read.
        name (TensorOrScalar): Optional operation name.

    Returns:
        TensorOrScalar: Tensor containing the raw file contents.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ReadFile", filename, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(
        "ReadFile",
        [filename],
        {"name": name},
        getattr(filename, "shape", ()),
        getattr(filename, "dtype", "float32"),
    )


def write_file(filename: str | Tensor, contents: Tensor, name: str | None = None) -> Tensor | None:
    """Write byte or string tensor contents to a file at the given path.

    Args:
        filename (str | Tensor): Target file path.
        contents (Tensor): Payload tensor to write.
        name (str | None): Optional operation name.

    Returns:
        Tensor | None: Symbolic operation node when tracing, or None in eager mode.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        import typing

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return typing.cast("Tensor | None", get_active_backend().execute_op("WriteFile", filename, contents, name=name))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("WriteFile", [filename, contents], {"name": name}, (), "float32")


def gfile_copy(src: str, dst: str, overwrite: bool = False) -> None:
    """Copy a file from source path to destination path.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.
        overwrite (bool): Whether to overwrite an existing destination file.

    Raises:
        FileExistsError: If `dst` already exists and `overwrite` is False.
    """
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"File {dst} already exists")
    shutil.copy2(src, dst)


def gfile_glob(pattern: str) -> list[str]:
    """Return a list of paths matching a pathname pattern.

    Args:
        pattern (str): Glob pattern string.

    Returns:
        list[str]: Matching file paths.
    """
    return glob.glob(pattern)


def gfile_stat(path: str) -> dict[str, int]:
    """Retrieve file size and modification timestamp metadata for a path.

    Args:
        path (str): Path to inspect.

    Returns:
        dict[str, int]: Dictionary with 'length' and 'mtime' integer fields.
    """
    st = os.stat(path)
    return {"length": st.st_size, "mtime": int(st.st_mtime)}


def gfile_makedirs(path: str) -> None:
    """Recursively create a directory and all intermediate parent directories.

    Args:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


@register_op("ReadFile")
class ReadFile(OpDef):
    """ReadFile operation definition."""

    op_name = "ReadFile"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for ReadFile.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


@register_op("WriteFile")
class WriteFile(OpDef):
    """WriteFile operation definition."""

    op_name = "WriteFile"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer broadcasted output shape for WriteFile.

        Args:
            *args (TensorOrScalar): Positional tensor arguments.
            **kwargs (TensorOrScalar): Keyword operation attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
