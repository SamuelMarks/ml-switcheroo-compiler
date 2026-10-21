"""Generic caching, archive extraction, progress bar, dtype standardization, and bounding-box utilities."""

from __future__ import annotations

import os
import random
import sys
import tarfile
import time
import typing
import urllib.request
import zipfile
from collections.abc import Iterable
from types import ModuleType

import pydantic

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, as_float, as_tensor
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.utils.serialization_utils import (
    CustomObjectScope,
    custom_object_scope,
    deserialize_object,
    get_custom_objects,
    get_registered_name,
    get_registered_object,
    register_serializable,
    serialize_object,
)

__all__ = [
    "ArchiveConfig",
    "CacheConfig",
    "CustomObjectScope",
    "GetFileConfig",
    "HashConfig",
    "Progbar",
    "ProgbarConfig",
    "ProgbarState",
    "bounding_boxes",
    "custom_object_scope",
    "deserialize_object",
    "disable_interactive_logging",
    "enable_interactive_logging",
    "get_custom_objects",
    "get_file",
    "get_registered_name",
    "get_registered_object",
    "is_interactive_logging_enabled",
    "is_symbolic_tensor",
    "register_serializable",
    "serialize_object",
    "set_random_seed",
    "standardize_dtype",
]


class HashConfig(pydantic.BaseModel):
    """Hash verification configuration for downloaded files."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    md5_hash: str | None = None
    file_hash: str | None = None
    hash_algorithm: str = "auto"


class ArchiveConfig(pydantic.BaseModel):
    """Archive extraction configuration for downloaded files."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    untar: bool = False
    extract: bool = False
    archive_format: str = "auto"


class CacheConfig(pydantic.BaseModel):
    """Local filesystem cache directory configuration."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cache_subdir: str = "datasets"
    cache_dir: str | None = None


class GetFileConfig(pydantic.BaseModel):
    """Composite configuration for `get_file` caching, hashing, and extraction."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    hash_config: HashConfig = pydantic.Field(default_factory=HashConfig)
    archive_config: ArchiveConfig = pydantic.Field(default_factory=ArchiveConfig)
    cache_config: CacheConfig = pydantic.Field(default_factory=CacheConfig)


class ProgbarConfig(pydantic.BaseModel):
    """Display configuration for `Progbar` CLI progress reporting."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    width: int = 30
    verbose: int = 1
    interval: float = 0.05
    stateful_metrics: list[str] | tuple[str, ...] | set[str] | None = None
    unit_name: str = "step"


def set_random_seed(seed: int) -> None:
    """Set global compiler, reference backend, and Python stdlib random seeds.

    Args:
        seed (int): Integer PRNG seed value.
    """
    config.seed = seed
    try:
        from ml_switcheroo_compiler.backends.numpy.utils import set_numpy_seed

        set_numpy_seed(seed)
    except (ImportError, AttributeError):
        _ = None
    try:
        random.seed(seed)
    except ImportError:
        _ = None


def _validate_cache(fpath: str) -> bool:
    """Check whether a cached file already exists at `fpath`.

    Args:
        fpath (str): Local file path to check.

    Returns:
        bool: True if `fpath` exists on disk.
    """
    return os.path.exists(fpath)


def _download_remote_file(origin: str, fpath: str) -> None:
    """Download a remote file from `origin` URL to local path `fpath`.

    Args:
        origin (str): Source URL to fetch.
        fpath (str): Destination local file path.

    Raises:
        RuntimeError: If downloading from `origin` fails.
    """
    try:
        urllib.request.urlretrieve(origin, fpath)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        raise RuntimeError(f"URL fetch failure on {origin}: {e} -- Please check your internet connection.") from e


def _extract_archive(fpath: str, datadir: str) -> None:
    """Extract a `.tar.gz`, `.tgz`, `.tar`, or `.zip` archive into `datadir`.

    Args:
        fpath (str): Path to the archive file.
        datadir (str): Destination extraction directory.
    """
    if fpath.endswith(".tar.gz") or fpath.endswith(".tgz"):
        with tarfile.open(fpath, "r:gz") as archive:
            archive.extractall(datadir)
    elif fpath.endswith(".tar"):
        with tarfile.open(fpath, "r:") as archive:
            archive.extractall(datadir)
    elif fpath.endswith(".zip"):
        with zipfile.ZipFile(fpath, "r") as archive:
            archive.extractall(datadir)


def get_file(
    fname: str,
    origin: str,
    config: GetFileConfig | None = None,
) -> str:
    """Download a file from `origin` URL if not already present in the local cache.

    Args:
        fname (str): Local filename under the cache subdirectory.
        origin (str): Remote URL from which to download the file.
        config (GetFileConfig | None): Optional cache, hash, and archive configuration.

    Returns:
        str: Local filesystem path to the downloaded (and optionally extracted) file.
    """
    conf = config if config is not None else GetFileConfig()
    untar = conf.archive_config.untar
    cache_subdir = conf.cache_config.cache_subdir
    extract = conf.archive_config.extract
    cache_dir = conf.cache_config.cache_dir

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".ml_switcheroo")

    datadir = os.path.join(cache_dir, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    if _validate_cache(fpath):
        return fpath

    _download_remote_file(origin, fpath)

    if untar or extract:
        _extract_archive(fpath, datadir)

    return fpath


class ProgbarState(pydantic.BaseModel):
    """Mutable runtime tracking state for a `Progbar` instance."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dynamic_display: bool
    total_width: int
    seen_so_far: int
    values: dict[str, list[float | int]]
    values_order: list[str]
    start_time: float
    last_update: float


class Progbar:
    """Command-line progress bar supporting stateful and stateless metric aggregation."""

    def __init__(
        self,
        target: int | None,
        config: ProgbarConfig | None = None,
    ) -> None:
        """Initialize `Progbar`.

        Args:
            target (int | None): Total number of steps expected, or None if unknown.
            config (ProgbarConfig | None): Optional display and metric configuration.
        """
        conf = config if config is not None else ProgbarConfig()

        self.target = target
        self.config = conf

        self.stateful_metrics = set(conf.stateful_metrics) if conf.stateful_metrics else set()
        self._values: dict[str, list[float | int]] = {}
        self._values_order: list[str] = []
        self._seen_so_far = 0
        self._last_update = 0.0

        self.state = ProgbarState(
            dynamic_display=hasattr(sys.stdout, "isatty") and sys.stdout.isatty(),
            total_width=0,
            seen_so_far=0,
            values=self._values,
            values_order=self._values_order,
            start_time=time.time(),
            last_update=0.0,
        )

    def _update_values(self, current: int, values: TensorOrScalar) -> None:
        """Update internal metric accumulators at step `current`.

        Args:
            current (int): Current step index.
            values (TensorOrScalar): Iterable of `(metric_name, metric_value)` pairs.
        """
        pairs: Iterable[tuple[TensorOrScalar, TensorOrScalar]] = typing.cast("Iterable[tuple[TensorOrScalar, TensorOrScalar]]", values) if isinstance(values, Iterable) else ()
        for item in pairs:
            k = str(item[0])
            v = float(as_float(item[1]))
            if k not in self._values_order:
                self._values_order.append(k)
            if k in self.stateful_metrics:
                self._values[k] = [v, 1]
            else:
                self._update_stateless_metric(k, v, current)

    def _update_stateless_metric(self, k: str, v: float, current: int) -> None:
        """Accumulate weighted running sum and weight for stateless metric `k`.

        Args:
            k (str): Metric identifier.
            v (float): Current step metric value.
            current (int): Step weight multiplier.
        """
        if k not in self._values:
            self._values[k] = [
                v * current,
                current,
            ]
        else:
            self._values[k][0] += v * current
            self._values[k][1] += current

    def _should_finalize(self, current: int, finalize: bool | None) -> bool:
        """Determine whether the progress bar has reached its final step.

        Args:
            current (int): Current step index.
            finalize (bool | None): Explicit finalize override flag, or None.

        Returns:
            bool: True if the progress bar should finalize.
        """
        if finalize is not None:
            return finalize
        return self.target is None or current >= self.target

    def _should_update(self, now: float, finalize: bool) -> bool:
        """Determine whether sufficient time has elapsed to refresh the display.

        Args:
            now (float): Current timestamp in seconds.
            finalize (bool): Whether this update is finalizing the progress bar.

        Returns:
            bool: True if the display should be updated.
        """
        return finalize or (now - self._last_update > self.config.interval)

    def _format_info(self, current: int) -> str:
        """Format step progress string for console output.

        Args:
            current (int): Current step index.

        Returns:
            str: Formatted step counter string.
        """
        return f" - {current}/{self.target}" if self.target is not None else f" - {current}"

    def update(self, current: int, values: TensorOrScalar = None, finalize: bool | None = None) -> None:
        """Advance the progress bar to step `current` and log metrics if due.

        Args:
            current (int): Current step count.
            values (TensorOrScalar): Optional sequence of `(name, value)` metric tuples.
            finalize (bool | None): Optional flag forcing completion of the progress bar.
        """
        values = values or []
        self._update_values(current, values)
        self._seen_so_far = current

        now = time.time()
        should_finalize = self._should_finalize(current, finalize)

        if self._should_update(now, should_finalize):
            self._last_update = now
            if self.config.verbose == 1:
                print(self._format_info(current))


def disable_interactive_logging(*args: str | int | bool | None, **kwargs: str | int | bool | None) -> None:
    """Disable interactive progress and diagnostic logging in the global environment.

    Args:
        *args (str | int | bool | None): Unused positional arguments.
        **kwargs (str | int | bool | None): Unused keyword arguments.
    """
    del args, kwargs
    config._state.env.interactive_logging = False


def enable_interactive_logging(*args: str | int | bool | None, **kwargs: str | int | bool | None) -> None:
    """Enable interactive progress and diagnostic logging in the global environment.

    Args:
        *args (str | int | bool | None): Unused positional arguments.
        **kwargs (str | int | bool | None): Unused keyword arguments.
    """
    del args, kwargs
    config._state.env.interactive_logging = True


def is_interactive_logging_enabled(*args: str | int | bool | None, **kwargs: str | int | bool | None) -> bool:
    """Check whether interactive logging is currently enabled.

    Args:
        *args (str | int | bool | None): Unused positional arguments.
        **kwargs (str | int | bool | None): Unused keyword arguments.

    Returns:
        bool: True if interactive logging is enabled, else False.
    """
    del args, kwargs
    return getattr(config._state.env, "interactive_logging", False)


def is_symbolic_tensor(*args: TensorOrScalar, **kwargs: AttrValue) -> bool:
    """Check whether the first positional argument is a symbolic compiler tensor.

    Args:
        *args (TensorOrScalar): Candidate object to inspect.
        **kwargs (AttrValue): Unused keyword arguments.

    Returns:
        bool: True if the object is a `Tensor`, `SymbolicTensor`, or `ProxyTensor`.
    """
    del kwargs
    if not args:
        return False
    x = args[0]
    type_name = type(x).__name__
    return type_name in ("Tensor", "SymbolicTensor", "ProxyTensor")


_DTYPE_ALIASES: dict[str, str] = {
    "float": "float32",
    "float32": "float32",
    "f4": "float32",
    "double": "float64",
    "float64": "float64",
    "f8": "float64",
    "int": "int32",
    "int32": "int32",
    "i4": "int32",
    "int64": "int64",
    "i8": "int64",
    "bool": "bool",
    "?": "bool",
}


def standardize_dtype(*args: TensorOrScalar, **kwargs: AttrValue) -> str | None:
    """Normalize a data type object or string alias into its canonical lowercase dtype name.

    Args:
        *args (TensorOrScalar): Data type object or string identifier.
        **kwargs (AttrValue): Unused keyword arguments.

    Returns:
        str | None: Canonical dtype name (e.g., `'float32'`, `'int64'`, `'bool'`), or None.
    """
    del kwargs
    if not args or args[0] is None:
        return None
    dtype = args[0]
    if hasattr(dtype, "name"):
        return str(dtype.name).lower()
    val_str = str(dtype).lower()
    if val_str.startswith("<class '") and val_str.endswith("'>"):
        val_str = val_str[8:-2]
    if "numpy." in val_str:
        val_str = val_str.split("numpy.")[-1]
    return _DTYPE_ALIASES.get(val_str, val_str)


class bounding_boxes:
    """Bounding box coordinate format validation and conversion utilities."""

    SUPPORTED_FORMATS: tuple[str, ...] = ("xyxy", "xywh", "yxyx", "center_xywh", "rel_xyxy")

    def __init__(self, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
        """Initialize `bounding_boxes` context wrapper.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (AttrValue): Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def validate_format(cls, box_format: str) -> None:
        """Validate that `box_format` is one of `SUPPORTED_FORMATS`.

        Args:
            box_format (str): Bounding box format identifier.

        Raises:
            ValueError: If `box_format` is not supported.
        """
        if str(box_format).lower() not in cls.SUPPORTED_FORMATS:
            msg = f"Unsupported bounding box format '{box_format}'. Expected one of {cls.SUPPORTED_FORMATS}"
            raise ValueError(msg)

    @staticmethod
    def _unpack_to_xyxy(
        arr: TensorOrScalar,
        source: str,
        src_lower: str,
        image_shape: tuple[int, int] | None,
    ) -> TensorOrScalar:
        """Unpack bounding box array from `source` format into `(x1, y1, x2, y2)` coordinate arrays.

        Args:
            arr (TensorOrScalar): Input bounding box array of shape `(..., 4)`.
            source (str): Original source format string.
            src_lower (str): Lowercase source format identifier.
            image_shape (tuple[int, int] | None): Optional `(height, width)` for relative coordinates.

        Returns:
            TensorOrScalar: 4-tuple of `(x1, y1, x2, y2)` coordinate arrays.

        Raises:
            ValueError: If relative coordinates are used without `image_shape` or format is unknown.
        """
        c0 = as_tensor(arr)[..., 0]
        c1 = as_tensor(arr)[..., 1]
        c2 = as_tensor(arr)[..., 2]
        c3 = as_tensor(arr)[..., 3]
        if src_lower == "xyxy":
            return c0, c1, c2, c3
        if src_lower == "xywh":
            return c0, c1, c0 + c2, c1 + c3
        if src_lower == "center_xywh":
            return c0 - c2 / 2.0, c1 - c3 / 2.0, c0 + c2 / 2.0, c1 + c3 / 2.0
        if src_lower == "yxyx":
            return c1, c0, c3, c2
        if src_lower == "rel_xyxy":
            if image_shape is None:
                msg = "image_shape (height, width) is required to convert from relative bounding box format."
                raise ValueError(msg)
            h_img, w_img = image_shape
            return c0 * w_img, c1 * h_img, c2 * w_img, c3 * h_img
        msg = f"Unrecognized source format '{source}'"
        raise ValueError(msg)

    @staticmethod
    def _pack_from_xyxy(
        backend_mod: ModuleType,
        arr: TensorOrScalar,
        coords: tuple[EagerTensorProtocol, ...],
        tgt_lower: str,
        image_shape: tuple[int, int] | None,
    ) -> TensorOrScalar:
        """Pack `(x1, y1, x2, y2)` coordinate arrays into the target bounding box format.

        Args:
            backend_mod (ModuleType): Array backend module providing `stack`.
            arr (TensorOrScalar): Original bounding box array fallback.
            coords (tuple[EagerTensorProtocol, ...]): Tuple of `(x1, y1, x2, y2)` coordinate arrays.
            tgt_lower (str): Lowercase target format name.
            image_shape (tuple[int, int] | None): Optional `(height, width)` for relative coordinates.

        Returns:
            TensorOrScalar: Stacked bounding box array in the target format.

        Raises:
            ValueError: If relative coordinates are requested without `image_shape`.
        """
        x1, y1, x2, y2 = coords
        if tgt_lower == "xyxy":
            return as_tensor(backend_mod.stack([x1, y1, x2, y2], axis=-1))
        if tgt_lower == "xywh":
            return as_tensor(backend_mod.stack([x1, y1, x2 - x1, y2 - y1], axis=-1))
        if tgt_lower == "center_xywh":
            return as_tensor(backend_mod.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1))
        if tgt_lower == "yxyx":
            return as_tensor(backend_mod.stack([y1, x1, y2, x2], axis=-1))
        if tgt_lower == "rel_xyxy":
            if image_shape is None:
                msg = "image_shape (height, width) is required to convert to relative bounding box format."
                raise ValueError(msg)
            h_img, w_img = image_shape
            return as_tensor(backend_mod.stack([x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img], axis=-1))
        return arr

    @classmethod
    def convert_format(
        cls,
        boxes: TensorOrScalar,
        source: str,
        target: str,
        image_shape: tuple[int, int] | None = None,
    ) -> TensorOrScalar:
        """Convert bounding boxes of shape `(..., 4)` from `source` format to `target` format.

        Args:
            boxes (TensorOrScalar): Bounding box coordinates array or sequence of shape `(..., 4)`.
            source (str): Source format name (`'xyxy'`, `'xywh'`, `'center_xywh'`, `'yxyx'`, `'rel_xyxy'`).
            target (str): Target format name (`'xyxy'`, `'xywh'`, `'center_xywh'`, `'yxyx'`, `'rel_xyxy'`).
            image_shape (tuple[int, int] | None): Optional `(height, width)` for relative coordinate scaling.

        Returns:
            TensorOrScalar: Converted bounding boxes array in `target` format.

        Raises:
            ValueError: If box shape or coordinate format is invalid.
        """
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        backend_mod = BackendRegistry.get("numpy").get_module()

        cls.validate_format(source)
        cls.validate_format(target)

        arr = backend_mod.asarray(boxes, dtype=backend_mod.float32)
        if arr.shape[-1] != 4:
            msg = f"Bounding boxes must have 4 coordinates in last dimension, got shape {arr.shape}"
            raise ValueError(msg)

        src_lower = source.lower()
        tgt_lower = target.lower()
        if src_lower == tgt_lower:
            return as_tensor(arr)

        coords = typing.cast(
            "tuple[EagerTensorProtocol, ...]",
            cls._unpack_to_xyxy(arr, source, src_lower, image_shape),
        )
        return cls._pack_from_xyxy(backend_mod, arr, coords, tgt_lower, image_shape)
