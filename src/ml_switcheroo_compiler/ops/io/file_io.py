# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""I/O and memory operations."""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def read_file(filename: str | Tensor, name: Any = None) -> Any:  # type: ignore
    """Read file.

    Args:
        filename (object): The filename parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ReadFile", filename, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ReadFile", [filename], {"name": name}, getattr(filename, "shape", ()), getattr(filename, "dtype", "float32"))


def write_file(filename: str | Tensor, contents: Tensor, name: Any = None) -> None:  # type: ignore
    """Write file.

    Args:
        filename (object): The filename parameter.
        contents (Tensor): The contents parameter.
        name (str): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("WriteFile", filename, contents, name=name)  # type: ignore
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("WriteFile", [filename, contents], {"name": name}, (), "float32")  # type: ignore


def gfile_copy(src: str, dst: str, overwrite: bool = False) -> None:
    """Copy a file.

    Args:
        src (str): The src parameter.
        dst (str): The dst parameter.
        overwrite (bool): The overwrite parameter.

    Raises:
        FileExistsError: An exception.
    """
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"File {dst} already exists")
    shutil.copy2(src, dst)


def gfile_glob(pattern: str) -> list[str]:
    """Glob pattern.

    Args:
        pattern (str): The pattern parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return glob.glob(pattern)


def gfile_stat(path: str) -> dict[str, int]:
    """Stat a file.

    Args:
        path (str): The path parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    st = os.stat(path)
    return {"length": st.st_size, "mtime": int(st.st_mtime)}


def gfile_makedirs(path: str) -> None:
    """Make directories.

    Args:
        path (str): The path parameter.
    """
    os.makedirs(path, exist_ok=True)


@register_op("ReadFile")
class ReadFile(OpDef):
    """ReadFile operation."""

    op_name = "ReadFile"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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
    """WriteFile operation."""

    op_name = "WriteFile"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res
