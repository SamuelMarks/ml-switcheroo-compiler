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


@register_op("SparsePlus")
class SparsePlus(OpDef):
    """SparsePlus operation."""

    op_name = "SparsePlus"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

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


@register_op("SparseSigmoid")
class SparseSigmoid(OpDef):
    """SparseSigmoid operation."""

    op_name = "SparseSigmoid"

    def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

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


def sparse_plus(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """SparsePlus frontend.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SparsePlus", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SparsePlus", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def sparse_sigmoid(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """SparseSigmoid frontend.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SparseSigmoid", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SparseSigmoid", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")
