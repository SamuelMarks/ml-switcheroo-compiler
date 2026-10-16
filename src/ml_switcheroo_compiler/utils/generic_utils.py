"""Provide generic utilities."""

from __future__ import annotations

# ruff: noqa: E402
import os
import random
import sys
import tarfile
import time
import typing  # noqa: F401
import urllib.request
import zipfile
from dataclasses import dataclass, field
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.utils.serialization_utils import (
    CustomObjectScope,  # noqa: F401
    custom_object_scope,  # noqa: F401
    deserialize_object,  # noqa: F401
    get_custom_objects,  # noqa: F401
    get_registered_name,  # noqa: F401
    get_registered_object,  # noqa: F401
    register_serializable,  # noqa: F401
    serialize_object,  # noqa: F401
)


@dataclass
class HashConfig:
    """Configuration class for hash config."""

    md5_hash: str | None = None
    file_hash: str | None = None
    hash_algorithm: str = "auto"


@dataclass
class ArchiveConfig:
    """Configuration class for archive config."""

    untar: bool = False
    extract: bool = False
    archive_format: str = "auto"


@dataclass
class CacheConfig:
    """Configuration class for cache config."""

    cache_subdir: str = "datasets"
    cache_dir: str | None = None


@dataclass
class GetFileConfig:
    """GetFile configuration."""

    hash_config: HashConfig = field(default_factory=HashConfig)
    archive_config: ArchiveConfig = field(default_factory=ArchiveConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class ProgbarConfig:
    """Progbar configuration."""

    width: int = 30
    verbose: int = 1
    interval: float = 0.05
    stateful_metrics: list[str] | tuple[str, ...] | set[str] | None = None
    unit_name: str = "step"


def set_random_seed(seed: int) -> None:
    """Set all random seeds for the program.

    Args:
        seed (int): The seed parameter.
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
    """Validate the cache.

    Args:
        fpath (str): The fpath parameter.

    Returns:
        bool: Result.
    """
    return os.path.exists(fpath)


def _download_remote_file(origin: str, fpath: str) -> None:
    """Download a remote file.

    Args:
        origin (str): The origin parameter.
        fpath (str): The fpath parameter.

    Raises:
        RuntimeError: An exception.
    """
    try:
        urllib.request.urlretrieve(origin, fpath)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        raise RuntimeError(f"URL fetch failure on {origin}: {e} -- Please check your internet connection.") from e


def _extract_archive(fpath: str, datadir: str) -> None:
    """Extract an archive.

    Args:
        fpath (str): The fpath parameter.
        datadir (str): The datadir parameter.
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
    """Download a file from a URL if it not already in the cache.

    Args:
        fname (str): The fname parameter.
        origin (str): The origin parameter.
        config (dict | list | str | bool | None): The config parameter.

    Returns:
        str: Result.
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


@dataclass
class ProgbarState:
    """Configuration class for progbar state."""

    dynamic_display: bool
    total_width: int
    seen_so_far: int
    values: dict[str, list[float | int]]
    values_order: list[str]
    start_time: float
    last_update: float


class Progbar:
    """Displays a progress bar."""

    def __init__(
        self,
        target: int | None,
        config: ProgbarConfig | None = None,
    ) -> None:
        """Initialize.

        Args:
            target (dict | list | str | bool | None): The target parameter.
            config (dict | list | str | bool | None): The config parameter.
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

    def _update_values(self, current: int, values: EagerValue) -> None:
        """Evaluate _update_values operation.

        Args:
            current (int): The current parameter.
            values (list): The values parameter.
        """
        for k, v in getattr(values, "__iter__")():
            if k not in self._values_order:
                self._values_order.append(k)
            if k in self.stateful_metrics:
                self._values[k] = [v, 1]
            else:
                self._update_stateless_metric(k, v, current)

    def _update_stateless_metric(self, k: str, v: float, current: int) -> None:
        """Evaluate _update_stateless_metric operation.

        Args:
            k (str): The k parameter.
            v (float): The v parameter.
            current (int): The current parameter.
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
        """Evaluate _should_finalize operation.

        Args:
        current (int): The current parameter.
        finalize (dict | list | str | bool | None): The finalize parameter.

        Returns:
        bool: Result.
        """
        if finalize is not None:
            return finalize
        return self.target is None or current >= self.target

    def _should_update(self, now: float, finalize: bool) -> bool:
        """Evaluate _should_update operation.

        Args:
        now (float): The now parameter.
        finalize (bool): The finalize parameter.

        Returns:
        bool: Result.
        """
        return finalize or (now - self._last_update > self.config.interval)

    def _format_info(self, current: int) -> str:
        """Evaluate _format_info operation.

        Args:
        current (int): The current parameter.

        Returns:
        str: Result.
        """
        return f" - {current}/{self.target}" if self.target is not None else f" - {current}"

    def update(self, current: int, values: EagerValue = None, finalize: bool | None = None) -> None:
        """Update the progress bar.

        Args:
            current (int): The current parameter.
            values (dict | list | str | bool | None): The values parameter.
            finalize (dict | list | str | bool | None): The finalize parameter.
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
    """Disable interactive logging.

    Args:
        *args: Arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    config._state.env.interactive_logging = False


def enable_interactive_logging(*args: str | int | bool | None, **kwargs: str | int | bool | None) -> None:
    """Enable interactive logging.

    Args:
        *args: Arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    config._state.env.interactive_logging = True


def is_interactive_logging_enabled(*args: str | int | bool | None, **kwargs: str | int | bool | None) -> bool:
    """Check if interactive logging is enabled.

    Args:
        *args: Arguments.
        **kwargs: Keyword arguments.

    Returns:
        bool: Whether interactive logging is enabled.
    """
    del args, kwargs
    return getattr(config._state.env, "interactive_logging", False)


def is_symbolic_tensor(*args: EagerValue, **kwargs: EagerValue) -> bool:
    """Check if an object is recognized as a compiler symbolic tensor.

    Args:
        *args: Target object to check.
        **kwargs: Keyword arguments.

    Returns:
        bool: Whether the object is recognized as a compiler symbolic tensor.
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


def standardize_dtype(*args: EagerValue, **kwargs: EagerValue) -> str | None:
    """Standardize a data type specification into its canonical lowercase string name.

    Args:
        *args: Data type object or string name.
        **kwargs: Keyword arguments.

    Returns:
        str | None: Canonical dtype string (e.g. 'float32', 'int64', 'bool') or None.
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
    """Bounding boxes utility class supporting coordinate format conversions and validations."""

    SUPPORTED_FORMATS: tuple[str, ...] = ("xyxy", "xywh", "yxyx", "center_xywh", "rel_xyxy")

    def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Initialize bounding boxes utility context.

        Args:
            *args: Positional args.
            **kwargs: Keyword args.
        """
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def validate_format(cls, box_format: str) -> None:
        """Validate if bounding box format is supported.

        Args:
            box_format (str): Bounding box format string.

        Raises:
            ValueError: If box format is not supported.
        """
        if str(box_format).lower() not in cls.SUPPORTED_FORMATS:
            msg = f"Unsupported bounding box format '{box_format}'. Expected one of {cls.SUPPORTED_FORMATS}"
            raise ValueError(msg)

    @staticmethod
    def _unpack_to_xyxy(arr: EagerValue, source: str, src_lower: str, image_shape: tuple[int, int] | None) -> EagerValue:
        """Unpack bounding box array from source format into (x1, y1, x2, y2) components.

        Args:
            arr: Input bounding box array.
            source: Original source format string.
            src_lower: Lowercase source format name.
            image_shape: Optional (height, width) tuple for relative coordinates.

        Returns:
            tuple: Unpacked (x1, y1, x2, y2) coordinate arrays.

        Raises:
            ValueError: If relative coordinates are used without image_shape or format is unrecognized.
        """
        c0, c1, c2, c3 = cast(EagerTensorProtocol, arr)[..., 0], cast(EagerTensorProtocol, arr)[..., 1], cast(EagerTensorProtocol, arr)[..., 2], cast(EagerTensorProtocol, arr)[..., 3]
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
    def _pack_from_xyxy(np: ModuleType, arr: EagerValue, coords: tuple[EagerTensorProtocol, ...], tgt_lower: str, image_shape: tuple[int, int] | None) -> EagerValue:
        """Pack (x1, y1, x2, y2) coordinate arrays into the target bounding box format.

        Args:
            np: NumPy backend module.
            arr: Original bounding box array fallback.
            coords: Tuple of (x1, y1, x2, y2) coordinate arrays.
            tgt_lower: Lowercase target format name.
            image_shape: Optional (height, width) tuple for relative coordinates.

        Returns:
            Stacked bounding box array in target format.

        Raises:
            ValueError: If relative coordinates are requested without image_shape.
        """
        x1, y1, x2, y2 = coords
        if tgt_lower == "xyxy":
            return cast(EagerValue, np.stack([x1, y1, x2, y2], axis=-1))
        if tgt_lower == "xywh":
            return cast(EagerValue, np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1))
        if tgt_lower == "center_xywh":
            return cast(EagerValue, np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1))
        if tgt_lower == "yxyx":
            return cast(EagerValue, np.stack([y1, x1, y2, x2], axis=-1))
        if tgt_lower == "rel_xyxy":
            if image_shape is None:
                msg = "image_shape (height, width) is required to convert to relative bounding box format."
                raise ValueError(msg)
            h_img, w_img = image_shape
            return cast(EagerValue, np.stack([x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img], axis=-1))
        return arr

    @classmethod
    def convert_format(
        cls,
        boxes: EagerValue,
        source: str,
        target: str,
        image_shape: tuple[int, int] | None = None,
    ) -> EagerValue:
        """Convert bounding boxes between coordinate formats.

        Args:
            boxes: Bounding box coordinates array or sequence of shape (..., 4).
            source (str): Source format name ('xyxy', 'xywh', 'center_xywh', etc.).
            target (str): Target format name ('xyxy', 'xywh', 'center_xywh', etc.).
            image_shape (tuple[int, int] | None): Optional (height, width) for relative coordinate scaling.

        Returns:
            Tensor: Converted bounding boxes array.

        Raises:
            ValueError: If box shape or format is invalid.
        """
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        np = BackendRegistry.get("numpy").get_module()

        cls.validate_format(source)
        cls.validate_format(target)

        arr = np.asarray(boxes, dtype=np.float32)
        if arr.shape[-1] != 4:
            msg = f"Bounding boxes must have 4 coordinates in last dimension, got shape {arr.shape}"
            raise ValueError(msg)

        src_lower = source.lower()
        tgt_lower = target.lower()
        if src_lower == tgt_lower:
            return cast(EagerValue, arr)

        coords = cls._unpack_to_xyxy(arr, source, src_lower, image_shape)
        return cast(EagerValue, getattr(cls, "_pack_from_xyxy")(np, arr, coords, tgt_lower, image_shape))
