# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""I/O and memory operations."""

from __future__ import annotations

import glob
import os
import shutil

from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import load_npz


def decode_csv(records: Tensor, record_defaults, field_delim=",", use_quote_delim=True, na_value="", select_cols=None, name=None) -> list[Tensor]:
    """Decode csv.

    Args:
        records (Tensor): The records parameter.
        record_defaults (list): The record_defaults parameter.
        field_delim (str): The field_delim parameter.
        use_quote_delim (bool): The use_quote_delim parameter.
        na_value (str): The na_value parameter.
        select_cols (list): The select_cols parameter.
        name (str): The name parameter.

    Returns:
        list: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DecodeCsv", records, record_defaults=record_defaults, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, select_cols=select_cols, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DecodeCsv", [records], {"record_defaults": record_defaults, "field_delim": field_delim, "use_quote_delim": use_quote_delim, "na_value": na_value, "select_cols": select_cols, "name": name}, getattr(records, "shape", ()), getattr(records, "dtype", "float32"))


def parse_example(serialized: Tensor, features, example_names=None, name=None) -> dict[str, Tensor]:
    """Parse example.

    Args:
        serialized (Tensor): The serialized parameter.
        features (dict): The features parameter.
        example_names (Tensor): The example_names parameter.
        name (str): The name parameter.

    Returns:
        dict: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseExample", serialized, features=features, example_names=example_names, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseExample", [serialized], {"features": features, "example_names": example_names, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


def serialize_tensor(tensor: Tensor, name=None):
    """Serialize tensor.

    Args:
        tensor (Tensor): The tensor parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SerializeTensor", tensor, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("SerializeTensor", [tensor], {"name": name}, getattr(tensor, "shape", ()), getattr(tensor, "dtype", "float32"))


def parse_tensor(serialized: Tensor, out_type: DType, name=None):
    """Parse tensor.

    Args:
        serialized (Tensor): The serialized parameter.
        out_type (DType): The out_type parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseTensor", serialized, out_type=out_type, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseTensor", [serialized], {"out_type": out_type, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


def parse_sequence_example(serialized: Tensor, context_features=None, sequence_features=None, example_names=None, name=None) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Parse sequence example.

    Args:
        serialized (Tensor): The serialized parameter.
        context_features (dict): The context_features parameter.
        sequence_features (dict): The sequence_features parameter.
        example_names (Tensor): The example_names parameter.
        name (str): The name parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("ParseSequenceExample", serialized, context_features=context_features, sequence_features=sequence_features, example_names=example_names, name=name)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("ParseSequenceExample", [serialized], {"context_features": context_features, "sequence_features": sequence_features, "example_names": example_names, "name": name}, getattr(serialized, "shape", ()), getattr(serialized, "dtype", "float32"))


class TFRecordOptions:
    """Options for TFRecordWriter."""

    def __init__(self, compression_type: str = "") -> None:
        """Initialize.

        Args:
            compression_type (str): The compression_type parameter.
        """
        self.compression_type = compression_type


class TFRecordWriter:
    """Writer for TFRecord format."""

    def __init__(self, path: str, options=None) -> None:
        """Initialize.

        Args:
            path (str): The path parameter.
            options (TFRecordOptions): The options parameter.
        """
        self.path = path
        self.options = options

    def write(self, record) -> None:
        """Write record.

        Args:
        record (object): The record parameter.

        Returns:
        NoneType: Result.
        """
        return None

    def close(self) -> None:
        """Close.

        Returns:
        NoneType: Result.
        """
        return None

    def __enter__(self) -> TFRecordWriter:
        """Enter context manager.

        Returns:
        TFRecordWriter: Result.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager.

        Args:
            exc_type (object): The exc_type parameter.
            exc_val (object): The exc_val parameter.
            exc_tb (object): The exc_tb parameter.
        """
        self.close()


@register_op("DecodeCsv")
class DecodeCsv(OpDef):
    """DecodeCsv operation."""

    op_name = "DecodeCsv"

    def infer_shape(self, *args, **kwargs):
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


@register_op("ParseExample")
class ParseExample(OpDef):
    """ParseExample operation."""

    op_name = "ParseExample"

    def infer_shape(self, *args, **kwargs):
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


@register_op("SerializeTensor")
class SerializeTensor(OpDef):
    """SerializeTensor operation."""

    op_name = "SerializeTensor"

    def infer_shape(self, *args, **kwargs):
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


@register_op("ParseTensor")
class ParseTensor(OpDef):
    """ParseTensor operation."""

    op_name = "ParseTensor"

    def infer_shape(self, *args, **kwargs):
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


@register_op("ParseSequenceExample")
class ParseSequenceExample(OpDef):
    """ParseSequenceExample operation."""

    op_name = "ParseSequenceExample"

    def infer_shape(self, *args, **kwargs):
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
