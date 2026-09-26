"""Unit tests providing 100% test, line, and branch coverage for base_generator.py."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.backends.base_generator import (
    BaseGenerator,
    CompiledArtifact,
)
from ml_switcheroo_compiler.ir.core import IRGraph


def test_compiled_artifact_edge_cases() -> None:
    """Test CompiledArtifact __getitem__, cleanup edge cases, execution, and attribute access."""
    # 1. Cleanup when temp_dir has no cleanup method
    artifact = CompiledArtifact(
        callable_fn=lambda x: x,
        binary_bytes=b"123",
        source_code="source",
        metadata={"custom_meta": 42},
        temp_dir="string_path_without_cleanup",
    )
    artifact.cleanup()
    assert artifact.temp_dir == "string_path_without_cleanup"

    # 1b. Execution when callable_fn is present and missing
    assert artifact(42) == 42

    artifact_no_fn = CompiledArtifact()
    with pytest.raises(RuntimeError, match="CompiledArtifact has no executable callable bound"):
        artifact_no_fn()

    # 1c. Cleanup when temp_dir has cleanup method
    class MockTempDir:
        """Mock temporary directory object with cleanup hook."""

        def __init__(self) -> None:
            """Initialize mock temp dir."""
            self.cleaned: bool = False

        def cleanup(self) -> None:
            """Mark cleaned as True."""
            self.cleaned = True

    mock_dir = MockTempDir()
    art_with_dir = CompiledArtifact(
        callable_fn=lambda: None,
        temp_dir=mock_dir,
    )
    art_with_dir.cleanup()
    assert mock_dir.cleaned is True
    assert art_with_dir.temp_dir is None

    # 2. __getitem__ accessing attribute
    assert artifact["binary_bytes"] == b"123"
    assert artifact["source_code"] == "source"
    assert artifact["custom_meta"] == 42

    # 3. __getitem__ key error
    with pytest.raises(KeyError):
        _ = artifact["nonexistent_attribute_or_meta"]


def test_base_generator_compile_aot_and_device_edge_cases() -> None:
    """Test BaseGenerator.compile_aot invocations, device discovery, and classmethods."""

    # 1. compile_aot called on a generator class directly (isinstance(self, type))
    class ConcreteAotGen(BaseGenerator):
        """Concrete generator for testing compile_aot on class."""

        def _compile_aot_impl(self, graph: IRGraph | None, **kwargs: object) -> str:
            """Compile AOT implementation."""
            del graph, kwargs
            return "compiled_output"

    g = IRGraph()
    res = ConcreteAotGen.compile_aot(g)
    assert res == "compiled_output"

    # 1b. compile_aot caching behavior on generator instance
    gen_inst = ConcreteAotGen(g)
    res_a = gen_inst.compile_aot(g, flag="test")
    res_b = gen_inst.compile_aot(g, flag="test")
    assert res_a is res_b

    # 2. compile_aot on instance without graph and graph=None
    class EmptyGen(BaseGenerator):
        """Empty generator without graph."""

        def __init__(self) -> None:
            """Initialize with None graph."""
            self.graph = None

    with pytest.raises(NotImplementedError):
        EmptyGen().compile_aot(None)

    # 3. Device discovery and classmethods
    class WebGPUGen(BaseGenerator):
        """WebGPU generator for device discovery testing."""

        pass

    class CudaGen(BaseGenerator):
        """CUDA generator for device discovery testing."""

        pass

    devices_default = BaseGenerator.get_logical_devices()
    assert len(devices_default) == 1

    devices_webgpu = WebGPUGen.get_logical_devices()
    assert len(devices_webgpu) == 2

    devices_cuda = CudaGen.get_logical_devices()
    assert len(devices_cuda) == 2

    # Device type filters
    assert len(CudaGen.get_logical_devices("gpu")) == 1
    assert len(CudaGen.get_logical_devices("cuda")) == 1
    assert len(CudaGen.get_logical_devices("cpu")) == 1
    assert len(CudaGen.get_logical_devices("nonexistent")) == 0

    assert len(BaseGenerator.get_physical_devices("cpu")) == 1
    mem_info = BaseGenerator.get_memory_info("gpu:0")
    assert mem_info == {"current": 0, "peak": 0}

    BaseGenerator.initialize_distributed()
    BaseGenerator.export_function()
