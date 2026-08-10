# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Distributed execution and sharding primitives."""

import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager as ContextManager
from typing import Any, Optional

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.ops import distributed_ops

from .device_mesh import DeviceMesh
from .layout_map import LayoutMap, ShardingSpec


class Distribution:
    """Base class for distributed strategies."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize Distribution.

        Args:
            *args: arguments.
            **kwargs: keyword arguments.
        """
        self.device_mesh = kwargs.get("device_mesh", None)

    def scope(self) -> ContextManager[None]:
        """Scope context manager.

        Returns:
            ContextManager[None]: the context manager scope.
        """

        @contextlib.contextmanager
        def _scope() -> Iterator[None]:
            r"""Activate this distribution strategy within the enclosing context block.

            Yields:
                None
            .
            """
            _old = _DIST_STATE["dist"]
            _DIST_STATE["dist"] = self  # type: ignore
            try:
                yield
            finally:
                _DIST_STATE["dist"] = _old

        return _scope()


class DataParallel(Distribution):
    """DataParallel strategy for distributed execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DataParallel distribution.

        Args:
            *args: arguments.
            **kwargs: keyword arguments.
        """
        super().__init__(*args, **kwargs)


class ModelParallel(Distribution):
    """ModelParallel strategy for distributed execution."""

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        layout_map: Optional[LayoutMap] = None,
        batch_dim_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the ModelParallel distribution.

        Args:
            device_mesh: The device mesh.
            layout_map: The layout map.
            batch_dim_name: Name of batch dimension.
            *args: Extra args.
            **kwargs: Extra kwargs.
        """
        self.layout_map = layout_map


class TensorLayoutClass:
    """Class representing TensorLayout."""

    def __init__(self, axes: tuple) -> None:
        """Initialize TensorLayoutClass.

        Args:
            axes (tuple): Tuple of axis names.
        """
        self.axes = axes


def TensorLayout(*args: Any, **kwargs: Any) -> Any:
    """Create a TensorLayout.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        TensorLayoutClass instance.
    """
    axes = kwargs.get("axes", args[0] if args else ())
    return TensorLayoutClass(axes)


def initialize(*args: Any, **kwargs: Any) -> None:
    """Initialize the distributed environment.

    Args:
        *args: positional args.
        **kwargs: keyword args.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend = registry.get_active_backend()
    if hasattr(backend, "initialize_distributed"):
        backend.initialize_distributed(*args, **kwargs)
    else:
        raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support initialize_distributed()")


def list_devices(*args: Any, **kwargs: Any) -> list:
    """List available devices.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        List of devices.
    """
    from ml_switcheroo_compiler.core.device import get_physical_devices

    return get_physical_devices()


_dist = None
_DIST_STATE: dict = {"dist": None}


def distribution(*args: Any, **kwargs: Any) -> Optional[Distribution]:
    """Get the current distribution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Optional[Distribution]: current active distribution.
    """
    return _DIST_STATE["dist"]


def set_distribution(dist: Distribution, *args: Any, **kwargs: Any) -> None:
    """Set the current distribution.

    Args:
        dist: Distribution to set.
        *args: arguments.
        **kwargs: keyword arguments.
    """
    _DIST_STATE["dist"] = dist


def distribute_tensor(*args: Any, **kwargs: Any) -> Any:
    """Distribute a tensor across the active distribution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns: Any: distributed tensor.
    """
    if _DIST_STATE["dist"] is None:
        return args[0] if args else kwargs.get("tensor")

    if config.eager_mode:
        return args[0] if args else kwargs.get("tensor")

    return distributed_ops.shard_tensor(*args, **kwargs)


__all__ = [
    "ShardingSpec",
]
