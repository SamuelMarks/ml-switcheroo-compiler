"""Behavioral unit tests for backend package inits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import ml_switcheroo_compiler.backends.awkward as awk_mod
import ml_switcheroo_compiler.backends.eager.core_math_ops as cmo_mod
import ml_switcheroo_compiler.backends.numpy as bnp_mod
import ml_switcheroo_compiler.backends.pyarrow_compute as pa_mod
import ml_switcheroo_compiler.ops.info_and_histograms as iah_mod
import ml_switcheroo_compiler.ops.text as text_mod
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.numpy import distributed as bnp_dist_mod
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from tests.testing_utils import GenericNDArray, adapt_backend_module


class _DummyGenerator:
    """Mock generator providing fallback prefix."""

    def get_fallback_prefix(self) -> str:
        """Return prefix for fallback ops.

        Returns:
            str: Prefix string.
        """
        return "np_eager"


class _MockArrayWithNumpy:
    """Mock object with numpy() method."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize with numpy array.

        Args:
            arr (GenericNDArray): Underlying array.
        """
        self.arr = arr

    def numpy(
        self,
    ) -> GenericNDArray:
        """Return underlying numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self.arr


class _MockDataWithNumpy:
    """Mock object with data.numpy() method."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize with nested data object.

        Args:
            arr (GenericNDArray): Underlying array.
        """
        self.data = _MockArrayWithNumpy(arr)


class _MockArrayWithToList:
    """Mock object with tolist() method."""

    def __init__(self, lst: list[float]) -> None:
        """Initialize with list.

        Args:
            lst (list[float]): List elements.
        """
        self.lst = lst

    def tolist(self) -> list[float]:
        """Return list representation.

        Returns:
            list[float]: The list.
        """
        return self.lst


class _NoShapeObject:
    """Mock object with dtype and tobytes, but missing shape."""

    dtype: str = "float32"

    def tobytes(self) -> bytes:
        """Return empty bytes.

        Returns:
            bytes: Empty bytes.
        """
        return b""


class _NoTobytesObject:
    """Mock object with dtype and shape, but missing tobytes."""

    dtype: str = "float32"
    shape: tuple[int, ...] = (2,)


class _MockBackendH5:
    """Mock backend implementing custom h5 save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: (
            tuple[
                dict[
                    str,
                    GenericNDArray,
                ],
                str,
            ]
            | None
        ) = None

    def load_h5(
        self, path: str
    ) -> dict[
        str,
        GenericNDArray,
    ]:
        """Load h5 mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([42.0])}

    def save_h5(
        self,
        weights: dict[
            str,
            GenericNDArray,
        ],
        path: str,
    ) -> None:
        """Save h5 mock weights.

        Args:
            weights (dict[str, GenericNDArray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)


class _MockBackendNPZ:
    """Mock backend implementing custom npz save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: (
            tuple[
                dict[
                    str,
                    GenericNDArray,
                ],
                str,
            ]
            | None
        ) = None

    def save_npz(
        self,
        weights: dict[
            str,
            GenericNDArray,
        ],
        path: str,
    ) -> None:
        """Save npz mock weights.

        Args:
            weights (dict[str, GenericNDArray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)

    def load_npz(
        self, path: str
    ) -> dict[
        str,
        GenericNDArray,
    ]:
        """Load npz mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([100.0])}


class _MockBackendSafetensors:
    """Mock backend implementing custom from_buffer method."""

    def from_buffer(
        self, buffer: bytes, dtype: str, shape: Sequence[int]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        """Reconstruct array from buffer.

        Args:
            buffer (bytes): Raw bytes.
            dtype (str): Dtype identifier.
            shape (Sequence[int]): Shape tuple or list.

        Returns:
            np.ndarray: Reconstructed array.
        """
        return np.frombuffer(buffer, dtype=np.float32).reshape(tuple(shape))


class _PlainBackend:
    """Plain backend without custom format hooks."""

    pass


def test_package_inits() -> None:
    """Verify package __init__.py modules and bound classmethods.

    Returns:
        None
    """
    assert awk_mod.AwkwardGenerator is AwkwardGenerator
    assert callable(awk_mod.AwkwardGenerator.zeros)
    assert callable(awk_mod.AwkwardGenerator.array)
    assert callable(awk_mod.AwkwardGenerator.get_layout)
    assert "AwkwardGenerator" in awk_mod.__all__

    assert pa_mod.PyArrowComputeGenerator is PyArrowComputeGenerator
    assert callable(pa_mod.PyArrowComputeGenerator.zeros)
    assert callable(pa_mod.PyArrowComputeGenerator.to_table)
    assert callable(pa_mod.PyArrowComputeGenerator.from_arrow)
    assert "PyArrowComputeGenerator" in pa_mod.__all__

    assert SparseGenerator.zeros((2, 2)).shape == (2, 2)
    assert SparseGenerator.array([1.0, 0.0], dtype="float32").shape == (2,)
    assert SparseGenerator.asarray([1.0, 0.0]).shape == (2,)
    assert SparseGenerator.item([3.5]) == 3.5
    assert SparseGenerator.execute_op("Add", SparseGenerator.array([1.0]), SparseGenerator.array([2.0])) is not None
    assert KerasCodeGenerator.execute_op("TupleGetItem", (10, 20), 0) == 10

    assert cmo_mod._accumulate_n is not None
    assert bnp_mod.NumpyGenerator is not None
    assert bnp_dist_mod.__doc__ is not None
    assert iah_mod.histogram is not None
    assert text_mod.AsStringConfig is not None


def test_edge_inits_symbol_exports() -> None:
    """Verify that canonical edge package and BackendRegistry export all edge generators."""
    import ml_switcheroo_compiler.backends.edge as edge
    import ml_switcheroo_compiler.backends.edge.distributed_webrtc as edge_webrtc
    import ml_switcheroo_compiler.backends.llvm_cpp as llvm_cpp
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    assert adapt_backend_module(edge).has_symbol("MLIRBytecodeEncoder")
    assert edge.MLIRBytecodeEncoder is not None
    assert adapt_backend_module(edge).has_symbol("ONNXCodeGenerator")
    assert edge.ONNXCodeGenerator is not None
    assert adapt_backend_module(edge).has_symbol("StableHLOCodeGenerator")
    assert edge.StableHLOCodeGenerator is not None
    assert adapt_backend_module(edge).has_symbol("WasmCodeGenerator")
    assert edge.WasmCodeGenerator is not None
    assert adapt_backend_module(edge).has_symbol("WebGLCodeGenerator")
    assert edge.WebGLCodeGenerator is not None
    assert adapt_backend_module(edge).has_symbol("WebGPUCodeGenerator")
    assert edge.WebGPUCodeGenerator is not None

    assert adapt_backend_module(edge_webrtc).has_symbol("SignalingHandler")
    assert edge_webrtc.SignalingHandler is not None
    assert adapt_backend_module(edge_webrtc).has_symbol("SignalingServer")
    assert edge_webrtc.SignalingServer is not None
    assert adapt_backend_module(edge_webrtc).has_symbol("WebRTCDataChannelStream")
    assert edge_webrtc.WebRTCDataChannelStream is not None

    assert adapt_backend_module(llvm_cpp).has_symbol("CppGenerator")
    assert llvm_cpp.CppGenerator is not None
    assert adapt_backend_module(llvm_cpp).has_symbol("LLVMCPPRunner")
    assert llvm_cpp.LLVMCPPRunner is not None

    assert BackendRegistry.get("edge") is edge.ONNXCodeGenerator
    assert BackendRegistry.get("edge_onnx") is edge.ONNXCodeGenerator
    assert BackendRegistry.get("edge_stablehlo") is edge.StableHLOCodeGenerator
    assert BackendRegistry.get("edge_wasm") is edge.WasmCodeGenerator
    assert BackendRegistry.get("edge_wasm_simd") is edge.WasmCodeGenerator
    assert BackendRegistry.get("edge_webgl") is edge.WebGLCodeGenerator
    assert BackendRegistry.get("edge_wgsl") is edge.WebGPUCodeGenerator
    assert BackendRegistry.get("webgpu") is edge.WebGPUCodeGenerator
