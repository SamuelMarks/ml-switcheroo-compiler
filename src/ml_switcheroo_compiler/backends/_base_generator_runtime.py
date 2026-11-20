"""AOT compilation, device introspection, and distributed runtime mixin for BaseGenerator."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.core.shape import OpValue as OpValue
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.core.device import Device

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends._base_generator_support import CompiledArtifact


class _CompileAOTMethodProtocol(typing.Protocol):
    """Protocol for ``compile_aot`` accepting either a generator instance or class."""

    def __call__(
        self,
        target: BaseGenerator | type[BaseGenerator],
        /,
        graph: IRGraph | None = None,
        **kwargs: BackendArg,
    ) -> CompiledArtifact:
        """Compile ``graph`` via ``target`` instance or class.

        Args:
            target (BaseGenerator | type[BaseGenerator]): Bound instance or generator class.
            graph (IRGraph | None): Optional target computation graph.
            **kwargs (BackendArg): Backend compiler options.

        Returns:
            CompiledArtifact: Standardized AOT artifact.
        """
        ...


class class_or_instance_method:
    """Descriptor supporting invocation on either a class or an instance."""

    def __init__(self, fn: _CompileAOTMethodProtocol) -> None:
        """Initialize descriptor with wrapped function.

        Args:
            fn (_CompileAOTMethodProtocol): The callable method to wrap.
        """
        self.fn: _CompileAOTMethodProtocol = fn

    def __get__(
        self,
        obj: BaseGenerator | None,
        cls: type[BaseGenerator] | None = None,
    ) -> TypedVariadicCallable[CompiledArtifact]:
        """Bind method to instance if present, else to class.

        Args:
            obj (BaseGenerator | None): Target instance or None.
            cls (type[BaseGenerator] | None): Owning class.

        Returns:
            TypedVariadicCallable[CompiledArtifact]: Bound callable.
        """
        import ml_switcheroo_compiler.backends.base_generator as _bg_mod

        target: BaseGenerator | type[BaseGenerator] = (
            obj if obj is not None else (cls if cls is not None else _bg_mod.BaseGenerator)
        )
        fn = self.fn

        def _bound(*args: BackendArg, **kwargs: BackendArg) -> CompiledArtifact:
            """Invoke bound method on `target` (instance or class).

            Args:
                *args (BackendArg): Positional arguments (optional ``IRGraph``).
                **kwargs (BackendArg): Keyword arguments.

            Returns:
                CompiledArtifact: Wrapped method result.
            """
            g_arg = args[0] if args and isinstance(args[0], IRGraph) else None
            return fn(target, g_arg, **kwargs)

        return _bound


class BaseGeneratorRuntimeMixin:
    """Mixin providing AOT compilation, device queries, and distributed execution hooks."""

    backend_name: str | None
    graph: IRGraph
    _aot_cache: dict[tuple[int, tuple[tuple[str, str], ...]], CompiledArtifact] | None
    generate: Callable[[], str]

    @class_or_instance_method
    def compile_aot(
        self: BaseGenerator | type[BaseGenerator], /, graph: IRGraph | None = None, **kwargs: BackendArg
    ) -> CompiledArtifact:
        """Compile an IRGraph into an ahead-of-time binary artifact or callable executable.

        Can be invoked either as an instance method (e.g. generator.compile_aot())
        or as a class method (e.g. GeneratorClass.compile_aot(graph)).

        Args:
            graph (IRGraph | None): Target computation graph to compile.
            **kwargs (BackendArg): Compiler options, optimization levels, or destination paths.

        Returns:
            CompiledArtifact: Compiled binary artifact, path, or callable execution wrapper.

        Raises:
            NotImplementedError: If AOT compilation is not implemented for this backend.
        """
        self_obj: BaseGenerator | type[BaseGenerator] = self
        raw_graph: IRGraph | None = (
            graph if graph is not None else (self_obj.graph if not isinstance(self_obj, type) else None)
        )
        target_graph: IRGraph = raw_graph if isinstance(raw_graph, IRGraph) else IRGraph()

        if isinstance(self, type):
            instance = self(target_graph)
            res_cls: CompiledArtifact = instance.compile_aot(target_graph, **kwargs)
            return res_cls

        if self._aot_cache is None:
            self._aot_cache = {}

        cache_key = (id(target_graph), tuple(sorted((str(k), str(v)) for k, v in kwargs.items())))
        if cache_key in self._aot_cache:
            return self._aot_cache[cache_key]

        raw_artifact = self._compile_aot_impl(target_graph, **kwargs)
        artifact: CompiledArtifact = (
            raw_artifact if isinstance(raw_artifact, CompiledArtifact) else CompiledArtifact(callable_fn=raw_artifact)
        )
        self._aot_cache[cache_key] = artifact
        return artifact

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Internal backend-specific AOT compilation hook.

        Args:
            graph (IRGraph): The computation graph to compile.
            **kwargs (BackendArg): Backend-specific compiler arguments.

        Returns:
            BackendArg: Resulting compiled artifact or callable.

        Raises:
            NotImplementedError: If `_compile_aot_impl` is not implemented by the subclass or target backend.
        """
        msg = f"AOT compilation is not implemented for {type(self).__name__}"
        raise NotImplementedError(msg)

    @classmethod
    def get_logical_devices(cls, device_type: str | None = None) -> list[Device]:
        """Discover logical devices available for this backend.

        Args:
            device_type (str | None): Optional device type filter (e.g., 'cpu', 'gpu', 'webgpu').

        Returns:
            list[Device]: Available logical devices.
        """
        from ml_switcheroo_compiler.core.device import Device, DeviceType

        devices: list[Device] = [Device(DeviceType.CPU, 0)]
        cls_name = cls.__name__.lower()
        if "webgpu" in cls_name or "wgsl" in cls_name:
            devices.append(Device(DeviceType.WEBGPU, 0))
        elif "cuda" in cls_name or "gpu" in cls_name or "rocm" in cls_name or "metal" in cls_name:
            devices.append(Device(DeviceType.GPU, 0))

        if device_type is not None:
            dtype_lower = str(device_type).lower()
            return [
                d
                for d in devices
                if (d.device_type if isinstance(d.device_type, str) else d.device_type.value) == dtype_lower
                or (dtype_lower in ("cuda", "rocm", "metal") and d.device_type == DeviceType.GPU)
            ]
        return devices

    @classmethod
    def get_physical_devices(cls, device_type: str | None = None) -> list[Device]:
        """Discover physical hardware accelerators available for this backend.

        Args:
            device_type (str | None): Optional device type filter.

        Returns:
            list[Device]: Available physical devices.
        """
        return cls.get_logical_devices(device_type)

    @classmethod
    def get_memory_info(cls, device: BackendArg = None) -> dict[str, int]:
        """Retrieve memory allocation statistics for the specified device.

        Args:
            device (BackendArg): Optional target device identifier.

        Returns:
            dict[str, int]: Dictionary containing 'current' and 'peak' memory in bytes.
        """
        del device
        return {"current": 0, "peak": 0}

    @classmethod
    def initialize_distributed(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Initialize distributed execution context for multi-device operations.

        Args:
            *args (BackendArg): Distributed initialization parameters.
            **kwargs (BackendArg): Additional distributed execution options.
        """
        del args, kwargs

    @classmethod
    def export_function(cls, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Export a computational graph or trace for unified persistence.

        Args:
            *args (BackendArg): Export arguments.
            **kwargs (BackendArg): Export configuration parameters.
        """
        del args, kwargs

    @classmethod
    def execute_collective(
        cls,
        op_name: str,
        tensor: BackendArg,
        **kwargs: BackendArg,
    ) -> BackendArg | None:
        """Execute an eager distributed collective on this backend if natively supported.

        Args:
            op_name (str): Canonical collective operation name.
            tensor (BackendArg): Input tensor buffer.
            **kwargs (BackendArg): Collective configuration parameters.

        Returns:
            BackendArg | None: Evaluated collective tensor, or ``None`` to fall back to host emulation.
        """
        del op_name, tensor, kwargs
        return None
