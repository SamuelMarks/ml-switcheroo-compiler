"""Distributed execution and sharding primitives."""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Iterator
from contextlib import AbstractContextManager as ContextManager

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.device import _backend_name
from ml_switcheroo_compiler.core.shape import IRAttrValue, OpValue, is_variadic_callable
from ml_switcheroo_compiler.ops import distributed_ops

from .device_mesh import DeviceMesh
from .layout_map import LayoutMap, ShardingSpec
from .sharding_lattice import (
    COMPILER_SHARDING_EQ,
    COMPILER_SHARDING_HASH,
    COMPILER_SHARDING_LATTICE,
    COMPILER_SHARDING_MONOID,
    COMPILER_SHARDING_PRETTY,
    ShardingLatticeElem,
    infer_collective_transition,
)


@typing.runtime_checkable
class _InitializeDistributedBackendProtocol(typing.Protocol):
    """Protocol for backends supporting ``initialize_distributed()``."""

    def initialize_distributed(self, *args: OpValue, **kwargs: IRAttrValue) -> None:
        """Initialize distributed backend.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (IRAttrValue): Keyword arguments.
        """
        ...


class Distribution:
    """Base class for distributed strategies."""

    def __init__(self, *args: OpValue, **kwargs: IRAttrValue) -> None:
        """Initialize Distribution.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (IRAttrValue): Keyword arguments.
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

    def __init__(self, *args: OpValue, **kwargs: IRAttrValue) -> None:
        """Initialize the DataParallel distribution.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (IRAttrValue): Keyword arguments.
        """
        super().__init__(*args, **kwargs)


class ModelParallel(Distribution):
    """ModelParallel strategy for distributed execution."""

    def __init__(
        self,
        device_mesh: DeviceMesh | None = None,
        layout_map: LayoutMap | None = None,
        batch_dim_name: str | None = None,
        *args: OpValue,
        **kwargs: IRAttrValue,
    ) -> None:
        """Initialize the ModelParallel distribution.

        Args:
            device_mesh (Optional[DeviceMesh]): The device mesh.
            layout_map (Optional[LayoutMap]): The layout map.
            batch_dim_name (Optional[str]): Name of batch dimension.
            *args (OpValue): Extra positional args.
            **kwargs (IRAttrValue): Extra keyword args.
        """
        _ = (device_mesh, batch_dim_name, args, kwargs)
        self.layout_map = layout_map


class TensorLayoutClass:
    """Class representing TensorLayout."""

    def __init__(self, axes: OpValue) -> None:
        """Initialize TensorLayoutClass.

        Args:
            axes (OpValue): Tuple of axis names.
        """
        self.axes = axes


def TensorLayout(*args: OpValue, **kwargs: IRAttrValue) -> TensorLayoutClass:
    """Create a TensorLayout.

    Args:
        *args (OpValue): Positional arguments.
        **kwargs (IRAttrValue): Keyword arguments.

    Returns:
        TensorLayoutClass: TensorLayoutClass instance.
    """
    axes = kwargs.get("axes", args[0] if args else ())
    return TensorLayoutClass(axes)


def initialize(*args: OpValue, **kwargs: IRAttrValue) -> None:
    """Initialize the distributed environment.

    Args:
        *args (OpValue): Positional args.
        **kwargs (IRAttrValue): Keyword args.

    Raises:
        BackendNotSupportedError: If the active backend does not support ``initialize_distributed()``.
    """
    import ml_switcheroo_compiler.backends.registry as registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    raw_backend = registry.get_active_backend()
    backend: OpValue | _InitializeDistributedBackendProtocol = raw_backend
    if isinstance(backend, _InitializeDistributedBackendProtocol) and callable(backend.initialize_distributed):
        backend.initialize_distributed(*args, **kwargs)
    else:
        raise BackendNotSupportedError(
            f"Active backend '{_backend_name(raw_backend)}' does not support initialize_distributed()"
        )


def list_devices(*args: OpValue, **kwargs: IRAttrValue) -> OpValue:
    """List available devices.

    Args:
        *args (OpValue): Positional arguments.
        **kwargs (IRAttrValue): Keyword arguments.

    Returns:
        OpValue: List of devices.
    """
    _ = (args, kwargs)
    from ml_switcheroo_compiler.core.device import get_physical_devices

    return get_physical_devices()


_dist: Distribution | None = None
_DIST_STATE: dict[str, Distribution | None] = {"dist": None}


def distribution(*args: OpValue, **kwargs: IRAttrValue) -> Distribution | None:
    """Get the current distribution.

    Args:
        *args (OpValue): Positional arguments.
        **kwargs (IRAttrValue): Keyword arguments.

    Returns:
        Optional[Distribution]: Current active distribution.
    """
    _ = (args, kwargs)
    return _DIST_STATE["dist"]


def set_distribution(dist: Distribution | None, *args: OpValue, **kwargs: IRAttrValue) -> None:
    """Set the current distribution.

    Args:
        dist (Distribution | None): Distribution to set.
        *args (OpValue): Positional arguments.
        **kwargs (IRAttrValue): Keyword arguments.
    """
    _ = (args, kwargs)
    _DIST_STATE["dist"] = dist


def distribute_tensor(*args: OpValue, **kwargs: IRAttrValue) -> OpValue:
    """Distribute a tensor across the active distribution.

    Args:
        *args (OpValue): Positional arguments.
        **kwargs (IRAttrValue): Keyword arguments.

    Returns:
        OpValue: Distributed tensor.
    """
    if _DIST_STATE["dist"] is None:
        return args[0] if args else kwargs.get("tensor")

    if config.eager_mode:
        return args[0] if args else kwargs.get("tensor")
    _callable_fn_1: OpValue = distributed_ops.shard_tensor
    assert is_variadic_callable(_callable_fn_1)
    return _callable_fn_1(*args, **kwargs)


__all__ = [
    "COMPILER_SHARDING_EQ",
    "COMPILER_SHARDING_HASH",
    "COMPILER_SHARDING_LATTICE",
    "COMPILER_SHARDING_MONOID",
    "COMPILER_SHARDING_PRETTY",
    "ShardingLatticeElem",
    "ShardingSpec",
    "infer_collective_transition",
]
