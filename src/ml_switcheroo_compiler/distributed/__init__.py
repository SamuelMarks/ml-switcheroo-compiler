"""Distributed execution and sharding primitives."""

import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager as ContextManager
from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.ops import distributed_ops

from .device_mesh import DeviceMesh
from .layout_map import LayoutMap, ShardingSpec


class Distribution:
    """Base class for distributed strategies."""

    def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
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
            _DIST_STATE["dist"] = self
            try:
                yield
            finally:
                _DIST_STATE["dist"] = _old

        return _scope()


class DataParallel(Distribution):
    """DataParallel strategy for distributed execution."""

    def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
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
        *args: EagerValue,
        **kwargs: EagerValue,
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

    def __init__(self, axes: EagerValue) -> None:
        """Initialize TensorLayoutClass.

        Args:
            axes (tuple): Tuple of axis names.
        """
        self.axes = axes


def TensorLayout(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Create a TensorLayout.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        TensorLayoutClass instance.
    """
    axes = kwargs.get("axes", args[0] if args else ())
    return cast(EagerValue, TensorLayoutClass(axes))


def initialize(*args: EagerValue, **kwargs: EagerValue) -> None:
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


def list_devices(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """List available devices.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        List of devices.
    """
    from ml_switcheroo_compiler.core.device import get_physical_devices

    return cast(EagerValue, get_physical_devices())


_dist: Optional[Distribution] = None
_DIST_STATE: dict[str, Optional[Distribution]] = {"dist": None}


def distribution(*args: EagerValue, **kwargs: EagerValue) -> Optional[Distribution]:
    """Get the current distribution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        Optional[Distribution]: current active distribution.
    """
    return _DIST_STATE["dist"]


def set_distribution(dist: Distribution, *args: EagerValue, **kwargs: EagerValue) -> None:
    """Set the current distribution.

    Args:
        dist: Distribution to set.
        *args: arguments.
        **kwargs: keyword arguments.
    """
    _DIST_STATE["dist"] = dist


def distribute_tensor(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Distribute a tensor across the active distribution.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns: Tensor: distributed tensor.
    """
    if _DIST_STATE["dist"] is None:
        return args[0] if args else kwargs.get("tensor")

    if config.eager_mode:
        return args[0] if args else kwargs.get("tensor")

    return cast(EagerValue, getattr(distributed_ops, "shard_tensor")(*args, **kwargs))


__all__ = [
    "ShardingSpec",
]
