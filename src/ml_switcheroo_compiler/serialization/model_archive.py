"""Zipped model archive serialization, loading, and archive editing."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Mapping

import pydantic

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.serialization.weights_io import (
    _compile_model_metadata,
    _extract_model_state,
    _extract_model_weights,
    _load_h5_weights,
    _save_as_h5,
)


def _write_h5_to_zip(zf: zipfile.ZipFile, filename: str, store: OpValue) -> None:
    """Write HDF5 data from a dictionary store into a zip file.

    Args:
        zf (zipfile.ZipFile): Input `zf` argument (zf).
        filename (str): Input `filename` argument (filename).
        store (OpValue): Input `store` argument (store).
    """
    zinfo = zipfile.ZipInfo(filename)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    _save_as_h5(store, tmp_path)
    with open(tmp_path, "rb") as f:
        zf.writestr(zinfo, f.read())
    os.remove(tmp_path)


class ArchiveSerializationContext(pydantic.BaseModel):
    """Provide context holding all stores and metadata for compiler archive serialization.

    Attributes:
        filepath (str): The destination file path.
        config_dict (dict[str, str | int | float | bool | list | dict] | None): The model configuration dictionary.
        metadata (dict[str, str | int | float | bool | list | dict] | None): The model metadata.
        weights_store (dict[str, OpValue] | None): The dictionary of extracted model weights.
        state_store (dict[str, OpValue] | None): The dictionary of extracted model state.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    filepath: str
    config_dict: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] | None = None
    metadata: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] | None = None
    weights_store: Mapping[str, Tensor | list[AttributeValue] | dict[str, AttributeValue] | OpValue] | None = None
    state_store: Mapping[str, Tensor | list[AttributeValue] | dict[str, AttributeValue] | OpValue] | None = None


def _write_archive_zip(ctx: ArchiveSerializationContext) -> None:
    """Write the collected stores to a zipped model archive file.

    Args:
        ctx (ArchiveSerializationContext): Archive serialization context.
    """
    with zipfile.ZipFile(ctx.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        if ctx.weights_store:
            _write_h5_to_zip(zf, "model.weights.h5", ctx.weights_store)
        if ctx.state_store:
            _write_h5_to_zip(zf, "optimizer.weights.h5", ctx.state_store)
        zf.writestr(zipfile.ZipInfo("config.json"), json.dumps(ctx.config_dict, indent=2))
        zf.writestr(zipfile.ZipInfo("metadata.json"), json.dumps(ctx.metadata, indent=2))


def save_model(
    model: OpValue, filepath: str, overwrite: bool = True, zipped: OpValue = None, **kwargs: OpValue
) -> None:
    """Save the model to a zipped compiler archive format, including state and weights.

    Args:
        model (OpValue): Input `model` argument (model).
        filepath (str): Input `filepath` argument (filepath).
        overwrite (bool): Input `overwrite` argument (overwrite).
        zipped (OpValue): Input `zipped` argument (zipped).
        **kwargs (OpValue): Keyword args.
    """
    _ = (overwrite, zipped, kwargs)
    config_dict, metadata = _compile_model_metadata(model)
    weights_store = _extract_model_weights(model)
    state_store = _extract_model_state(model, weights_store)
    ctx = ArchiveSerializationContext(
        filepath=filepath,
        config_dict=config_dict,
        metadata=metadata,
        weights_store=weights_store,
        state_store=state_store,
    )
    _write_archive_zip(ctx)


class LoadedModel:
    """LoadedModel executable model container holding configuration, weights, and metadata."""

    def __init__(
        self,
        config: dict[str, AttributeValue],
        metadata: dict[str, AttributeValue] | None = None,
        weights: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize LoadedModel instance.

        Args:
            config (dict[str, AttributeValue]): Model architecture configuration dictionary.
            metadata (Optional[dict[str, AttributeValue]]): Model serialization metadata dictionary.
            weights (Optional[dict[str, AttributeValue]]): Restored model weights dictionary.
        """
        self.config = config
        self.metadata = metadata or {}
        self.weights = weights or {}


def _read_h5_from_zip(zf: zipfile.ZipFile, filename: str) -> dict[str, OpValue]:
    """Read HDF5 weights from an entry inside an open zip archive.

    Args:
        zf (zipfile.ZipFile): Open zip archive.
        filename (str): Entry name inside the zip archive.

    Returns:
        dict[str, OpValue]: Loaded weights dictionary.
    """
    if filename not in zf.namelist():
        return {}
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(zf.read(filename))
        raw_weights = _load_h5_weights(tmp_path)
        return dict(raw_weights) if isinstance(raw_weights, dict) else {}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_model(
    filepath: str,
    custom_objects: dict[str, AttributeValue] | None = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: AttributeValue,
) -> LoadedModel:
    """Load and reconstruct a serialized model archive from a zipped archive file.

    Args:
        filepath (str): Path to the model archive file.
        custom_objects (Optional[dict[str, AttributeValue]]): Mapping of custom classes or functions.
        compile (bool): Whether to compile the model after loading.
        safe_mode (bool): Whether to disallow arbitrary code execution.
        **kwargs (AttributeValue): Additional keyword arguments.

    Returns:
        LoadedModel: Fully reconstructed model container with config, metadata, and weights.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If archive is corrupted or not a valid zip archive.
    """
    _ = (custom_objects, compile, safe_mode, kwargs)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found")

    if os.path.isdir(filepath):
        raise ValueError(f"Invalid model archive: '{filepath}' is a directory, expected a model zip archive")

    if not zipfile.is_zipfile(filepath):
        raise ValueError(f"Invalid or corrupted model archive at '{filepath}'")

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            if "config.json" not in zf.namelist():
                raise ValueError(f"Model archive '{filepath}' is missing 'config.json'")
            config = json.loads(zf.read("config.json").decode("utf-8"))
            metadata = json.loads(zf.read("metadata.json").decode("utf-8")) if "metadata.json" in zf.namelist() else {}
            weights = _read_h5_from_zip(zf, "model.weights.h5")
            return LoadedModel(config=config, metadata=metadata, weights=weights)
    except (ValueError, FileNotFoundError):
        raise
    except (OSError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        raise ValueError(f"Failed to load model from '{filepath}': {e}") from e


class ArchiveFileEditor:
    """Editor class for interacting with and modifying serialized model archive files."""

    def __init__(self, filepath: str) -> None:
        """Initialize ArchiveFileEditor.

        Args:
            filepath (str): The destination file path.
        """
        self.filepath = filepath
