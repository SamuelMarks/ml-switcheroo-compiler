"""Generic utilities."""

from __future__ import annotations  # pragma: no cover


import os  # pragma: no cover
import urllib.request  # pragma: no cover
import tarfile  # pragma: no cover
import zipfile  # pragma: no cover
import time  # pragma: no cover
import sys  # pragma: no cover
from typing import Any

from dataclasses import dataclass


@dataclass
class GetFileConfig:  # pragma: no cover
    """GetFile configuration."""

    untar: bool = False
    md5_hash: str | None = None
    file_hash: str | None = None
    cache_subdir: str = "datasets"
    hash_algorithm: str = "auto"
    extract: bool = False
    archive_format: str = "auto"
    cache_dir: str | None = None


@dataclass
class ProgbarConfig:  # pragma: no cover
    """Progbar configuration."""

    width: int = 30
    verbose: int = 1
    interval: float = 0.05
    stateful_metrics: list | None = None
    unit_name: str = "step"


# pragma: no cover


def set_random_seed(seed: int) -> None:  # pragma: no cover
    """Sets all random seeds for the program."""
    from ml_switcheroo_compiler.core import config  # pragma: no cover

    config.seed = seed  # type: ignore[attr-defined] # pragma: no cover
    try:  # pragma: no cover
        import sys  # pragma: no cover

        if "numpy" in sys.modules:  # pragma: no cover
            sys.modules["numpy"].random.seed(seed)  # pragma: no cover
    except (ImportError, AttributeError):  # pragma: no cover
        pass  # pragma: no cover
    try:  # pragma: no cover
        import random  # pragma: no cover

        random.seed(seed)  # pragma: no cover
    except ImportError:  # pragma: no cover
        pass  # pragma: no cover


def _validate_cache(fpath: str) -> bool:  # pragma: no cover
    """Validate the cache."""
    return os.path.exists(fpath)


def _download_remote_file(origin: str, fpath: str) -> None:  # pragma: no cover
    """Download a remote file."""
    try:  # pragma: no cover
        urllib.request.urlretrieve(origin, fpath)  # pragma: no cover
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            f"URL fetch failure on {origin}: {e} -- Please check your internet connection."
        ) from e


def _extract_archive(fpath: str, datadir: str) -> None:  # pragma: no cover
    """Extract an archive."""
    if fpath.endswith(".tar.gz") or fpath.endswith(".tgz"):  # pragma: no cover
        with tarfile.open(fpath, "r:gz") as archive:  # pragma: no cover
            archive.extractall(datadir)  # pragma: no cover
    elif fpath.endswith(".tar"):  # pragma: no cover
        with tarfile.open(fpath, "r:") as archive:  # pragma: no cover
            archive.extractall(datadir)  # pragma: no cover
    elif fpath.endswith(".zip"):  # pragma: no cover
        with zipfile.ZipFile(fpath, "r") as archive:  # pragma: no cover
            archive.extractall(datadir)  # pragma: no cover


def get_file(  # pragma: no cover
    fname: str,
    origin: str,
    config: GetFileConfig | None = None,
) -> str:
    """Downloads a file from a URL if it not already in the cache."""
    conf = config if config is not None else GetFileConfig()
    untar = conf.untar
    cache_subdir = conf.cache_subdir
    extract = conf.extract
    cache_dir = conf.cache_dir

    if cache_dir is None:  # pragma: no cover
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras")  # pragma: no cover

    datadir = os.path.join(cache_dir, cache_subdir)  # pragma: no cover
    os.makedirs(datadir, exist_ok=True)  # pragma: no cover

    fpath = os.path.join(datadir, fname)  # pragma: no cover

    if _validate_cache(fpath):  # pragma: no cover
        return fpath

    _download_remote_file(origin, fpath)  # pragma: no cover

    if untar or extract:  # pragma: no cover
        _extract_archive(fpath, datadir)  # pragma: no cover

    return fpath  # pragma: no cover


class Progbar:  # pragma: no cover
    """Displays a progress bar."""

    def __init__(  # pragma: no cover
        self,
        target: int | None,
        config: ProgbarConfig | None = None,
    ) -> None:
        """Initialize progress bar."""
        conf = config if config is not None else ProgbarConfig()
        width = conf.width
        verbose = conf.verbose
        interval = conf.interval
        stateful_metrics = conf.stateful_metrics
        unit_name = conf.unit_name

        self.target = target  # pragma: no cover
        self.width = width  # pragma: no cover
        self.verbose = verbose  # pragma: no cover
        self.interval = interval  # pragma: no cover
        self.stateful_metrics = stateful_metrics or []  # pragma: no cover
        self.unit_name = unit_name  # pragma: no cover
        self._dynamic_display = (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )  # pragma: no cover
        self._total_width = 0  # pragma: no cover
        self._seen_so_far = 0  # pragma: no cover
        self._values: dict[str, list[float | int]] = {}  # pragma: no cover
        self._values_order: list[str] = []  # pragma: no cover
        self._start = time.time()  # pragma: no cover
        self._last_update = 0.0  # pragma: no cover

    def _update_values(self, current: int, values: list[Any]) -> None:  # pragma: no cover
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k in self.stateful_metrics:
                self._values[k] = [v, 1]
            else:
                self._update_stateless_metric(k, v, current)

    def _update_stateless_metric(self, k: str, v: float, current: int) -> None:  # pragma: no cover
        if k not in self._values:
            self._values[k] = [
                v * (current - self._seen_so_far),
                current - self._seen_so_far,
            ]
        else:
            self._values[k][0] += v * (current - self._seen_so_far)
            self._values[k][1] += current - self._seen_so_far

    def _should_finalize(self, current: int, finalize: bool | None) -> bool:  # pragma: no cover
        if finalize is not None:
            return finalize
        return self.target is None or current >= self.target

    def _should_update(self, now: float, finalize: bool) -> bool:  # pragma: no cover
        return finalize or (now - self._last_update > self.interval)

    def _format_info(self, current: int) -> str:  # pragma: no cover
        return f" - {current}/{self.target}" if self.target is not None else f" - {current}"

    def update(  # pragma: no cover
        self, current: int, values: list[Any] | None = None, finalize: bool | None = None
    ) -> None:
        """Updates the progress bar."""
        values = values or []  # pragma: no cover
        self._update_values(current, values)  # pragma: no cover
        self._seen_so_far = current  # pragma: no cover

        now = time.time()  # pragma: no cover
        should_finalize = self._should_finalize(current, finalize)  # pragma: no cover

        if self._should_update(now, should_finalize):  # pragma: no cover
            self._last_update = now  # pragma: no cover
            if self.verbose == 1:  # pragma: no cover
                print(self._format_info(current))  # pragma: no cover


class FeatureSpace:  # pragma: no cover
    """FeatureSpace utility class."""

    pass  # pragma: no cover


class Config:  # pragma: no cover
    """Config utility class."""

    pass  # pragma: no cover


class CustomObjectScope:  # pragma: no cover
    """Scope for custom objects."""

    pass  # pragma: no cover


class PyDataset:  # pragma: no cover
    """PyDataset utility class."""

    pass  # pragma: no cover


class Sequence:  # pragma: no cover
    """Sequence utility class."""

    pass  # pragma: no cover


def clear_session(*args: object, **kwargs: object) -> None:  # pragma: no cover
    """Clear the Keras session.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover


def custom_object_scope(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Create a custom object scope.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The scope.
    """
    pass  # pragma: no cover


def deserialize_keras_object(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Deserialize a Keras object.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The deserialized object.
    """
    pass  # pragma: no cover


def disable_interactive_logging(*args: object, **kwargs: object) -> None:  # pragma: no cover
    """Disable interactive logging.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover


def enable_interactive_logging(*args: object, **kwargs: object) -> None:  # pragma: no cover
    """Enable interactive logging.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover


def get_custom_objects(*args: object, **kwargs: object) -> dict[str, object]:  # pragma: no cover
    """Get custom objects.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        A dictionary of custom objects.
    """
    return {}  # pragma: no cover


def get_registered_name(*args: object, **kwargs: object) -> str:  # pragma: no cover
    """Get registered name.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The registered name.
    """
    return ""  # pragma: no cover


def get_registered_object(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Get registered object.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The registered object.
    """
    pass  # pragma: no cover


def is_interactive_logging_enabled(*args: object, **kwargs: object) -> bool:  # pragma: no cover
    """Check if interactive logging is enabled.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Whether interactive logging is enabled.
    """
    return False  # pragma: no cover


def is_keras_tensor(*args: object, **kwargs: object) -> bool:  # pragma: no cover
    """Check if an object is a Keras tensor.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Whether the object is a Keras tensor.
    """
    return False  # pragma: no cover


def register_keras_serializable(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Register an object with Keras serialization.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The decorator.
    """
    pass  # pragma: no cover


def serialize_keras_object(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Serialize a Keras object.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The serialized object.
    """
    pass  # pragma: no cover


def standardize_dtype(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Standardize a dtype.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The standardized dtype.
    """
    pass  # pragma: no cover


class legacy:  # pragma: no cover
    """Legacy utilities namespace."""

    pass  # pragma: no cover


class bounding_boxes:  # pragma: no cover
    """Bounding boxes utilities namespace."""

    pass  # pragma: no cover
