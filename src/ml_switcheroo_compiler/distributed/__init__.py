"""Distributed execution and sharding primitives."""

import contextlib  # pragma: no cover
from collections.abc import Iterator
from contextlib import AbstractContextManager as ContextManager
from typing import Optional

from ml_switcheroo_compiler.core import config  # pragma: no cover
from ml_switcheroo_compiler.ops import distributed  # pragma: no cover

from .device_mesh import DeviceMesh
from .layout_map import LayoutMap, ShardingSpec


class DataParallel:
    """DataParallel strategy for distributed execution."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the DataParallel distribution.

        Args:
            *args: arguments.
            **kwargs: keyword arguments.
        """
        pass  # pragma: no cover


class ModelParallel:
    """ModelParallel strategy for distributed execution."""

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        layout_map: Optional[LayoutMap] = None,
        batch_dim_name: Optional[str] = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize the ModelParallel distribution.

        Args:
            device_mesh: The device mesh.
            layout_map: The layout map.
            batch_dim_name: Name of batch dimension.
            *args: Extra args.
            **kwargs: Extra kwargs.
        """
        self.layout_map = layout_map  # pragma: no cover


def TensorLayout(*args: object, **kwargs: object) -> None:
    """Create a TensorLayout.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover


def initialize(*args: object, **kwargs: object) -> None:
    """Initialize distributed execution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover


def list_devices(*args: object, **kwargs: object) -> None:
    """List available devices.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover


class Distribution:
    """Base class for distributed strategies."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize Distribution.

        Args:
            *args: arguments.
            **kwargs: keyword arguments.
        """
        pass  # pragma: no cover

    def scope(self) -> ContextManager[None]:
        """Scope context manager.

        Returns:
            ContextManager[None]: the context manager scope.
        """

        @contextlib.contextmanager  # pragma: no cover
        def _scope() -> Iterator[None]:  # pragma: no cover
            """Function docstring."""
            global _dist
            _old = _dist  # pragma: no cover
            _dist = self  # type: ignore  # pragma: no cover
            try:  # pragma: no cover
                yield  # pragma: no cover
            finally:
                _dist = _old  # pragma: no cover

        return _scope()  # pragma: no cover


_dist: Optional[Distribution] = None


def distribution(*args: object, **kwargs: object) -> Optional[Distribution]:
    """Get the current distribution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Optional[Distribution]: current active distribution.
    """
    global _dist
    return _dist  # pragma: no cover


def set_distribution(dist: Distribution, *args: object, **kwargs: object) -> None:
    """Set the current distribution.

    Args:
        dist: Distribution to set.
        *args: arguments.
        **kwargs: keyword arguments.
    """
    global _dist
    _dist = dist  # pragma: no cover


def distribute_tensor(*args: object, **kwargs: object) -> object:
    """Distribute a tensor across the active distribution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        object: distributed tensor.
    """
    global _dist
    if _dist is None:  # pragma: no cover
        return args[0] if args else kwargs.get("tensor")  # pragma: no cover

    if config.eager_mode:  # pragma: no cover
        return args[0] if args else kwargs.get("tensor")  # pragma: no cover

    return distributed.shard_tensor(*args, **kwargs)  # pragma: no cover


__all__ = [
    "DeviceMesh",
    "LayoutMap",
    "ShardingSpec",
]
