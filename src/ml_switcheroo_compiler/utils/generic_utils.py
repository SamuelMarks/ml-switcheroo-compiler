"""Module generic_utils.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Provide generic utilities."""


import os
import random
import sys
import tarfile
import time
import typing
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Any

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.serialization import custom_object_scope, deserialize_keras_object, get_custom_objects, get_registered_name, get_registered_object, register_keras_serializable, serialize_keras_object


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
    stateful_metrics: Any | None = None
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
        config (Any): The config parameter.

    Returns:
        str: Result.
    """
    conf = config if config is not None else GetFileConfig()
    untar = conf.archive_config.untar
    cache_subdir = conf.cache_config.cache_subdir
    extract = conf.archive_config.extract
    cache_dir = conf.cache_config.cache_dir

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras")

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
            target (Any): The target parameter.
            config (Any): The config parameter.
        """
        conf = config if config is not None else ProgbarConfig()

        self.target = target
        self.config = conf

        self.stateful_metrics = set(conf.stateful_metrics) if conf.stateful_metrics else set()
        self._values = {}
        self._values_order = []
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

    def _update_values(self, current: int, values) -> None:
        """Evaluate _update_values operation.

        Args:
            current (int): The current parameter.
            values (list): The values parameter.
        """
        for k, v in values:
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
        finalize (Any): The finalize parameter.

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

    def update(self, current: int, values=None, finalize: bool | None = None) -> None:
        """Update the progress bar.

        Args:
            current (int): The current parameter.
            values (Any): The values parameter.
            finalize (Any): The finalize parameter.
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


class FeatureSpace:
    """FeatureSpace utility class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class Config:
    """Config utility class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class PyDataset:
    """PyDataset utility class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class Sequence:
    """Sequence utility class."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


_GLOBAL_CUSTOM_OBJECTS: dict[str, object] = {}
_REGISTERED_NAMES: dict[object, str] = {}


class CustomObjectScope:
    """Context manager for temporary registration of custom objects."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize CustomObjectScope with dictionary or kwargs of custom objects.

        Args:
            *args (object): Mapping dictionary of custom objects.
            **kwargs (object): Key-value pairs of custom objects.
        """
        self.custom_objects: dict[str, object] = {}
        if args and isinstance(args[0], dict):
            self.custom_objects.update(args[0])
        self.custom_objects.update(kwargs)
        self.backup: dict[str, object] = {}

    def __enter__(self) -> CustomObjectScope:
        """Enter scope and merge custom objects into global registry.

        Returns:
            CustomObjectScope: The active scope context manager.
        """
        self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
        _GLOBAL_CUSTOM_OBJECTS.update(self.custom_objects)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit scope and restore previous global custom object state.

        Args:
            exc_type (object): Exception type.
            exc_val (object): Exception value.
            exc_tb (object): Exception traceback.
        """
        _GLOBAL_CUSTOM_OBJECTS.clear()
        _GLOBAL_CUSTOM_OBJECTS.update(self.backup)


def custom_object_scope(*args: object, **kwargs: object) -> CustomObjectScope:
    """Create a functional custom object scope context manager.

    Args:
        *args (object): Mapping dictionary of custom objects.
        **kwargs (object): Custom objects key-value pairs.

    Returns:
        CustomObjectScope: The context manager instance.
    """
    return CustomObjectScope(*args, **kwargs)


custom_Any_scope = custom_object_scope


def get_custom_objects() -> dict[str, object]:
    """Retrieve the global registry of custom objects.

    Returns:
        dict[str, object]: Dictionary of registered custom objects.
    """
    return _GLOBAL_CUSTOM_OBJECTS


get_custom_Anys = get_custom_objects


def get_registered_name(obj: object = None) -> str:
    """Get the registered serializable name of a class or function.

    Args:
        obj (object): Target class or function.

    Returns:
        str: Registered name string or empty string.
    """
    if obj is None:
        return ""
    if obj in _REGISTERED_NAMES:
        return _REGISTERED_NAMES[obj]
    if hasattr(obj, "__name__"):
        return str(obj.__name__)
    return str(type(obj).__name__)


def get_registered_object(name: str | None = None, custom_objects: dict[str, object] | None = None) -> object | None:
    """Retrieve registered object by name from custom objects or global registry.

    Args:
        name (str | None): Identifier name of the object.
        custom_objects (dict[str, object] | None): Optional custom objects to search first.

    Returns:
        object | None: Found class/function or None.
    """
    if not name:
        return None
    if custom_objects and name in custom_objects:
        return custom_objects[name]
    if name in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[name]
    return None


get_registered_Any = get_registered_object


def register_keras_serializable(package: str = "Custom", name: str | None = None) -> typing.Callable[[object], object]:
    """Register a class or function as Keras serializable.

    Args:
        package (str): Package namespace name. Defaults to "Custom".
        name (str | None): Optional explicit registration name.

    Returns:
        typing.Callable[[object], object]: Class/function decorator.
    """

    def decorator(cls_or_fn: object) -> object:
        """Register the annotated class in the keras registry.

        Args:
            cls_or_fn (object): The class or function to register.

        Returns:
            object: The original class or function.
        """
        registered_name = name or getattr(cls_or_fn, "__name__", str(cls_or_fn))
        full_name = f"{package}>{registered_name}" if package else registered_name
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = cls_or_fn
        _GLOBAL_CUSTOM_OBJECTS[full_name] = cls_or_fn
        _REGISTERED_NAMES[cls_or_fn] = registered_name
        return cls_or_fn

    return decorator


def serialize_keras_object(obj: object = None) -> dict[str, object] | None:
    """Serialize a Keras object or layer into a configuration dictionary.

    Args:
        obj (object): Object to serialize.

    Returns:
        dict[str, object] | None: Serialized dictionary or None.
    """
    if obj is None:
        return {}
    cls_name = get_registered_name(obj.__class__)
    cfg: dict[str, object] = obj.get_config() if hasattr(obj, "get_config") else {}
    return {
        "class_name": cls_name,
        "config": cfg,
        "module": getattr(obj.__class__, "__module__", ""),
        "registered_name": cls_name,
    }


serialize_keras_Any = serialize_keras_object


def deserialize_keras_object(
    identifier: object = None,
    custom_objects: dict[str, object] | None = None,
) -> object:
    """Deserialize a Keras configuration dictionary back into an object instance.

    Args:
        identifier (object): Serialized config dict or object.
        custom_objects (dict[str, object] | None): Optional custom objects map.

    Returns:
        object: Deserialized object instance.
    """
    if identifier is None:
        return {}
    if not isinstance(identifier, dict):
        return identifier
    class_name = identifier.get("class_name")
    if not isinstance(class_name, str):
        return identifier
    config_dict = identifier.get("config", {})
    cls_obj = get_registered_object(class_name, custom_objects)
    if cls_obj is not None and isinstance(cls_obj, type):
        if hasattr(cls_obj, "from_config") and isinstance(config_dict, dict):
            return cls_obj.from_config(config_dict)
        if isinstance(config_dict, dict):
            return cls_obj(**config_dict)
    return identifier


deserialize_keras_Any = deserialize_keras_object


def clear_session(*args: object, **kwargs: object) -> None:
    """Clear the Keras session.

    Args:
        *args (object): Arguments.
        **kwargs (object): Keyword arguments.
    """
    del args, kwargs
    config.clear_cache()


def disable_interactive_logging(*args: object, **kwargs: object) -> None:
    """Disable interactive logging.

    Args:
        *args (object): Arguments.
        **kwargs (object): Keyword arguments.
    """
    del args, kwargs
    config._state.env.interactive_logging = False


def enable_interactive_logging(*args: object, **kwargs: object) -> None:
    """Enable interactive logging.

    Args:
        *args (object): Arguments.
        **kwargs (object): Keyword arguments.
    """
    del args, kwargs
    config._state.env.interactive_logging = True


def is_interactive_logging_enabled(*args: object, **kwargs: object) -> bool:
    """Check if interactive logging is enabled.

    Args:
        *args (object): Arguments.
        **kwargs (object): Keyword arguments.

    Returns:
        bool: Whether interactive logging is enabled.
    """
    del args, kwargs
    return getattr(config._state.env, "interactive_logging", False)


def is_keras_tensor(*args: object, **kwargs: object) -> bool:
    """Check if an object is a Keras tensor or compiler symbolic tensor.

    Args:
        *args (object): Target object to check.
        **kwargs (object): Keyword arguments.

    Returns:
        bool: Whether the object is recognized as a Keras/symbolic tensor.
    """
    del kwargs
    if not args:
        return False
    x = args[0]
    if hasattr(x, "_keras_history") or getattr(x, "is_keras_tensor", False):
        return True
    type_name = type(x).__name__
    return type_name in ("KerasTensor", "Tensor", "SymbolicTensor")


def standardize_dtype(*args: object, **kwargs: object) -> str | None:
    """Standardize a data type specification into its canonical lowercase string name.

    Args:
        *args (object): Data type object or string name.
        **kwargs (object): Keyword arguments.

    Returns:
        str | None: Canonical dtype string (e.g. 'float32', 'int64', 'bool') or None.
    """
    del kwargs
    if not args:
        return None
    dtype = args[0]
    if dtype is None:
        return None
    if hasattr(dtype, "name"):
        return str(dtype.name).lower()
    val_str = str(dtype).lower()
    if val_str.startswith("<class '") and val_str.endswith("'>"):
        val_str = val_str[8:-2]
    if "numpy." in val_str:
        val_str = val_str.split("numpy.")[-1]
    if val_str in ("float", "float32", "f4"):
        return "float32"
    if val_str in ("double", "float64", "f8"):
        return "float64"
    if val_str in ("int", "int32", "i4"):
        return "int32"
    if val_str in ("int64", "i8"):
        return "int64"
    if val_str in ("bool", "?"):
        return "bool"
    return val_str


class bounding_boxes:
    """Bounding boxes utility class supporting coordinate format conversions and validations."""

    SUPPORTED_FORMATS: tuple[str, ...] = ("xyxy", "xywh", "yxyx", "center_xywh", "rel_xyxy")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize bounding boxes utility context.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
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

    @classmethod
    def convert_format(
        cls,
        boxes: object,
        source: str,
        target: str,
        image_shape: tuple[int, int] | None = None,
    ) -> object:
        """Convert bounding boxes between coordinate formats.

        Args:
            boxes (object): Bounding box coordinates array or sequence of shape (..., 4).
            source (str): Source format name ('xyxy', 'xywh', 'center_xywh', etc.).
            target (str): Target format name ('xyxy', 'xywh', 'center_xywh', etc.).
            image_shape (tuple[int, int] | None): Optional (height, width) for relative coordinate scaling.

        Returns:
            object: Converted bounding boxes array.

        Raises:
            ValueError: If box shape or format is invalid.
        """
        import numpy as np

        cls.validate_format(source)
        cls.validate_format(target)

        arr = np.asarray(boxes, dtype=np.float32)
        if arr.shape[-1] != 4:
            msg = f"Bounding boxes must have 4 coordinates in last dimension, got shape {arr.shape}"
            raise ValueError(msg)

        src_lower = source.lower()
        tgt_lower = target.lower()
        if src_lower == tgt_lower:
            return arr

        if src_lower == "xyxy":
            x1, y1, x2, y2 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        elif src_lower == "xywh":
            x1, y1, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            x2, y2 = x1 + w, y1 + h
        elif src_lower == "center_xywh":
            cx, cy, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0
        elif src_lower == "yxyx":
            y1, x1, y2, x2 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        elif src_lower == "rel_xyxy":
            if image_shape is None:
                msg = "image_shape (height, width) is required to convert from relative bounding box format."
                raise ValueError(msg)
            h_img, w_img = image_shape
            x1 = arr[..., 0] * w_img
            y1 = arr[..., 1] * h_img
            x2 = arr[..., 2] * w_img
            y2 = arr[..., 3] * h_img
        else:
            msg = f"Unrecognized source format '{source}'"
            raise ValueError(msg)

        if tgt_lower == "xyxy":
            return np.stack([x1, y1, x2, y2], axis=-1)
        if tgt_lower == "xywh":
            return np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)
        if tgt_lower == "center_xywh":
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            return np.stack([cx, cy, w, h], axis=-1)
        if tgt_lower == "yxyx":
            return np.stack([y1, x1, y2, x2], axis=-1)
        if tgt_lower == "rel_xyxy":
            if image_shape is None:
                msg = "image_shape (height, width) is required to convert to relative bounding box format."
                raise ValueError(msg)
            h_img, w_img = image_shape
            return np.stack([x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img], axis=-1)

        return arr
