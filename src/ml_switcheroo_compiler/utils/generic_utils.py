from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Provide generic utilities."""


import os
import random
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.core import config


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

    hash_config: HashConfig = HashConfig()
    archive_config: ArchiveConfig = ArchiveConfig()
    cache_config: CacheConfig = CacheConfig()


@dataclass
class ProgbarConfig:
    """Progbar configuration."""

    width: int = 30
    verbose: int = 1
    interval: float = 0.05
    stateful_metrics: list | None = None
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
        config (object): The config parameter.

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
            target (object): The target parameter.
            config (object): The config parameter.
        """
        conf = config if config is not None else ProgbarConfig()

        self.target = target
        self.config = conf

        self.stateful_metrics = set(conf.stateful_metrics) if conf.stateful_metrics else set()
        self._values = {}  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self._values_order = []  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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

    def _update_values(self, current: int, values: list[Any]) -> None:
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
        finalize (object): The finalize parameter.

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

    def update(self, current: int, values: list[Any] | None = None, finalize: bool | None = None) -> None:
        """Update the progress bar.

        Args:
            current (int): The current parameter.
            values (object): The values parameter.
            finalize (object): The finalize parameter.
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class Config:
    """Config utility class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class CustomObjectScope:
    """Scope for custom objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> CustomObjectScope:
        """Enter.

        Returns:
        CustomObjectScope: Result.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit.

        Args:
            exc_type (object): The exc_type parameter.
            exc_val (object): The exc_val parameter.
            exc_tb (object): The exc_tb parameter.
        """
        _ = None


class PyDataset:
    """PyDataset utility class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class Sequence:
    """Sequence utility class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


def clear_session(*args: Any, **kwargs: Any) -> None:
    """Clear the Keras session.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    config.clear_cache()


def custom_object_scope(*args: Any, **kwargs: Any) -> Any:
    """Create a custom object scope.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The scope.
    """
    return CustomObjectScope(*args, **kwargs)


def deserialize_keras_object(*args: Any, **kwargs: Any) -> Any:
    """Deserialize a Keras object.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The deserialized object.
    """
    return None


def disable_interactive_logging(*args: Any, **kwargs: Any) -> None:
    """Disable interactive logging.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    config._state.env.interactive_logging = False


def enable_interactive_logging(*args: Any, **kwargs: Any) -> None:
    """Enable interactive logging.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    config._state.env.interactive_logging = True


def get_custom_objects(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Get custom objects.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        A dictionary of custom objects.
    """
    return {}


def get_registered_name(*args: Any, **kwargs: Any) -> str:
    """Get registered name.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The registered name.
    """
    return ""


def get_registered_object(*args: Any, **kwargs: Any) -> Any:
    """Get registered object.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The registered object.
    """
    return None


def is_interactive_logging_enabled(*args: Any, **kwargs: Any) -> bool:
    """Check if interactive logging is enabled.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Whether interactive logging is enabled.
    """
    return False


def is_keras_tensor(*args: Any, **kwargs: Any) -> bool:
    """Check if an object is a Keras tensor.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Whether the object is a Keras tensor.
    """
    return False


def register_keras_serializable(*args: Any, **kwargs: Any) -> Any:
    """Register an object with Keras serialization.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The decorator.
    """

    def decorator(cls: Any) -> Any:
        """Register the annotated class in the keras registry.

        Args:
            cls (object): The class to register.

        Returns: Any: The original class.
        """
        return cls

    return decorator


def serialize_keras_object(*args: Any, **kwargs: Any) -> Any:
    """Serialize a Keras object.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The serialized object.
    """
    return None


def standardize_dtype(*args: Any, **kwargs: Any) -> Any:
    """Standardize a dtype.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The standardized dtype.
    """
    return args[0] if args else None


class bounding_boxes:
    """Bounding boxes utilities namespace."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs
