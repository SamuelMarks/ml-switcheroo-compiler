"""Distributed execution and sharding primitives."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager as ContextManager

from ml_switcheroo_compiler.backends.eager_registry import as_tensor
from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.ops import distributed_ops

from .device_mesh import DeviceMesh
from .layout_map import LayoutMap, ShardingSpec


class Distribution:
    """Base class for distributed strategies."""

    def __init__(self, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
        """Initialize Distribution.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (AttrValue): Keyword arguments.
        """
        _ = args
        self.device_mesh = kwargs.get("device_mesh", None)

    def scope(self) -> ContextManager[None]:
        """Scope context manager.

        Returns:
            ContextManager[None]: The context manager scope.
        """

        @contextlib.contextmanager
        def _scope() -> Iterator[None]:
            """Activate this distribution strategy within the enclosing context block.

            Yields:
                None: Enclosing context block.
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

    def __init__(self, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
        """Initialize the DataParallel distribution.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (AttrValue): Keyword arguments.
        """
        super().__init__(*args, **kwargs)


class ModelParallel(Distribution):
    """ModelParallel strategy for distributed execution."""

    def __init__(
        self,
        device_mesh: DeviceMesh | None = None,
        layout_map: LayoutMap | None = None,
        batch_dim_name: str | None = None,
        *args: TensorOrScalar,
        **kwargs: AttrValue,
    ) -> None:
        """Initialize the ModelParallel distribution.

        Args:
            device_mesh (Optional[DeviceMesh]): The device mesh.
            layout_map (Optional[LayoutMap]): The layout map.
            batch_dim_name (Optional[str]): Name of batch dimension.
            *args (TensorOrScalar): Extra positional args.
            **kwargs (AttrValue): Extra keyword args.
        """
        _ = (device_mesh, batch_dim_name, args, kwargs)
        self.layout_map = layout_map


class TensorLayoutClass:
    """Class representing TensorLayout."""

    def __init__(self, axes: TensorOrScalar) -> None:
        """Initialize TensorLayoutClass.

        Args:
            axes (TensorOrScalar): Tuple of axis names.
        """
        self.axes = axes


def TensorLayout(*args: TensorOrScalar, **kwargs: AttrValue) -> TensorLayoutClass:
    """Create a TensorLayout.

    Args:
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.

    Returns:
        TensorLayoutClass: TensorLayoutClass instance.
    """
    axes = kwargs.get("axes", args[0] if args else ())
    return TensorLayoutClass(axes)


def initialize(*args: TensorOrScalar, **kwargs: AttrValue) -> None:
    """Initialize the distributed environment.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (AttrValue): Keyword args.

    Raises:
        BackendNotSupportedError: If the active backend does not support initialize_distributed().
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    backend = registry.get_active_backend()
    if hasattr(backend, "initialize_distributed"):
        backend.initialize_distributed(*args, **kwargs)
    else:
        raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support initialize_distributed()")


def list_devices(*args: TensorOrScalar, **kwargs: AttrValue) -> TensorOrScalar:
    """List available devices.

    Args:
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.

    Returns:
        TensorOrScalar: List of devices.
    """
    _ = (args, kwargs)
    from ml_switcheroo_compiler.core.device import get_physical_devices

    return as_tensor(get_physical_devices())


_dist: Distribution | None = None
_DIST_STATE: dict[str, Distribution | None] = {"dist": None}


def distribution(*args: TensorOrScalar, **kwargs: AttrValue) -> Distribution | None:
    """Get the current distribution.

    Args:
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.

    Returns:
        Optional[Distribution]: Current active distribution.
    """
    _ = (args, kwargs)
    return _DIST_STATE["dist"]


def set_distribution(dist: Distribution, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
    """Set the current distribution.

    Args:
        dist (Distribution): Distribution to set.
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.
    """
    _ = (args, kwargs)
    _DIST_STATE["dist"] = dist


def distribute_tensor(*args: TensorOrScalar, **kwargs: AttrValue) -> TensorOrScalar:
    """Distribute a tensor across the active distribution.

    Args:
        *args (TensorOrScalar): Positional arguments.
        **kwargs (AttrValue): Keyword arguments.

    Returns:
        TensorOrScalar: Distributed tensor.
    """
    if _DIST_STATE["dist"] is None:
        return args[0] if args else kwargs.get("tensor")

    if config.eager_mode:
        return args[0] if args else kwargs.get("tensor")

    return as_tensor(getattr(distributed_ops, "shard_tensor")(*args, **kwargs))


__all__ = [
    "ShardingSpec",
]
