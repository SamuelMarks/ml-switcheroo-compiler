"""Behavioral tests for backend generator delegates, symbolic collective/upsample/stack fallbacks, and math_misc ops."""

from __future__ import annotations

import types
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.cuda.cuda import _as_tpl_opt
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.wasm import _as_wasm_attr_val
from ml_switcheroo_compiler.backends.hardware_config_models import HardwareTemplateConfig
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.numba import kernels as numba_kernels
from ml_switcheroo_compiler.backends.numpy.eager.indexing import gather_eager
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.distributed.host_collectives import reduce_scatter
from ml_switcheroo_compiler.ops.control_flow.eager import _map_fn_eager_get_length
from ml_switcheroo_compiler.ops.info_and_histograms import math_misc
from ml_switcheroo_compiler.ops.nn.upsample_ops import upsample_nearest
from ml_switcheroo_compiler.ops.shape.joining import _resolve_stack_dtype_str, stack
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import MockTensorStub, MockTestStub, ensure_tensor, require_callable


class _StubDaskArr:
    """Minimal Dask array stub supporting `.compute()` and `__array__`."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None:
        """Initialize with backing ndarray.

        Args:
            arr (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Backing NumPy array.
        """
        self._arr = arr

    def compute(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Return materialized NumPy array.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Materialized array.
        """
        return self._arr

    def __array__(
        self, dtype: np.dtype[np.float64] | None = None, copy: bool | None = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Convert stub to NumPy ndarray.

        Args:
            dtype (np.dtype[np.float64] | None): Optional target dtype.
            copy (bool | None): Optional NumPy 2.0 copy flag.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Underlying ndarray.
        """
        _ = copy
        return np.asarray(self._arr, dtype=dtype)


def _build_fallback_backend_stubs() -> tuple[types.ModuleType, dict[str, types.ModuleType]]:
    """Construct lightweight Dask/MLX/Torch module stubs when optional runtimes are uninstalled.

    Returns:
        tuple[types.ModuleType, dict[str, types.ModuleType]]: `(fake_da, sys_modules_patch_dict)`.
    """
    import importlib.util
    import types

    mp = pytest.MonkeyPatch()
    fake_da = types.ModuleType("dask.array")
    mp.setattr(fake_da, "add", lambda a, b: np.add(np.asarray(a), np.asarray(b)), raising=False)
    mp.setattr(
        fake_da,
        "zeros",
        lambda shape, dtype=None, chunks="auto": _StubDaskArr(np.zeros(shape, dtype=dtype)),
        raising=False,
    )
    mp.setattr(
        fake_da,
        "array",
        lambda data, dtype=None: _StubDaskArr(
            np.array(data._arr if isinstance(data, _StubDaskArr) else data, dtype=dtype)
        ),
        raising=False,
    )
    mp.setattr(
        fake_da,
        "from_array",
        lambda data, dtype=None, chunks="auto": _StubDaskArr(
            np.array(data._arr if isinstance(data, _StubDaskArr) else data, dtype=dtype)
        ),
        raising=False,
    )
    mp.setattr(
        fake_da,
        "asarray",
        lambda data, dtype=None: _StubDaskArr(
            np.asarray(data._arr if isinstance(data, _StubDaskArr) else data, dtype=dtype)
        ),
        raising=False,
    )
    stub_sys: dict[str, types.ModuleType] = {}
    if importlib.util.find_spec("dask") is None:
        fake_dask = types.ModuleType("dask")
        fake_dask.__path__ = []
        mp.setattr(fake_dask, "array", fake_da, raising=False)
        stub_sys["dask"] = fake_dask
        stub_sys["dask.array"] = fake_da
    if importlib.util.find_spec("mlx") is None:
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__path__ = []
        fake_mx_core = types.ModuleType("mlx.core")
        mp.setattr(fake_mlx, "core", fake_mx_core, raising=False)
        stub_sys["mlx"] = fake_mlx
        stub_sys["mlx.core"] = fake_mx_core
    if importlib.util.find_spec("torch") is None:
        stub_sys["torch"] = types.ModuleType("torch")
    return fake_da, stub_sys


def test_backend_generator_array_and_execution_delegates() -> None:
    """Verify CuPy, Dask, MLX, PyTorch, Numba, CUDA, and WASM runtime delegates."""
    import ml_switcheroo_compiler.ops.generated as gen_pkg

    assert gen_pkg.__name__ == "ml_switcheroo_compiler.ops.generated"

    tpl_cfg = _as_tpl_opt({"body": "out[idx] = in0[idx];"})
    assert isinstance(tpl_cfg, HardwareTemplateConfig)
    assert tpl_cfg.body == "out[idx] = in0[idx];"
    assert _as_tpl_opt("invalid_template") is None

    wasm_dict = _as_wasm_attr_val({"alpha": 2, "beta": 3.5})
    assert wasm_dict == {"alpha": 2, "beta": 3.5}

    with (
        patch("ml_switcheroo_compiler.backends.cupy.types.cp", np),
        patch("ml_switcheroo_compiler.backends.cupy.eager.cp", np),
    ):
        cp_z = CupyGenerator.zeros((2, 2))
        np.testing.assert_allclose(np.asarray(cp_z), np.zeros((2, 2)))
        cp_a = CupyGenerator.array([1.0, 2.0])
        cp_b = CupyGenerator.asarray([3.0, 4.0])
        assert CupyGenerator.item(CupyGenerator.array(4.5)) == 4.5
        cp_sum = CupyGenerator.execute_op("Add", cp_a, cp_b)
        np.testing.assert_allclose(np.asarray(cp_sum), np.array([4.0, 6.0]))

    import sys

    from ml_switcheroo_compiler.backends.dask import eager as dask_eager_mod
    from ml_switcheroo_compiler.backends.dask import types as dask_types_mod

    fake_da, stub_sys = _build_fallback_backend_stubs()
    dask_patch = (
        patch.object(dask_types_mod, "da", fake_da)
        if dask_types_mod.da is None
        else patch.object(dask_types_mod, "da", dask_types_mod.da)
    )
    dask_eager_patch = (
        patch.object(dask_eager_mod, "da", fake_da)
        if dask_eager_mod.da is None
        else patch.object(dask_eager_mod, "da", dask_eager_mod.da)
    )

    with dask_patch, dask_eager_patch, patch.dict(sys.modules, stub_sys):
        da_z = DaskGenerator.zeros((2, 2))
        np.testing.assert_allclose(np.asarray(da_z), np.zeros((2, 2)))
        da_a = DaskGenerator.array([1.0, 2.0])
        da_b = DaskGenerator.asarray([5.0, 6.0])
        assert DaskGenerator.item(DaskGenerator.array(7.5)) == 7.5
        da_sum = DaskGenerator.execute_op("Add", da_a, da_b)
        np.testing.assert_allclose(np.asarray(da_sum), np.array([6.0, 8.0]))

        assert MLXCodeGenerator.get_module().__name__ == "mlx.core"
        assert PyTorchCodeGenerator.get_module().__name__ == "torch"

    with ConfigContext(backend="numpy", eager_mode=True):
        nb_abs = require_callable(numba_kernels.abs)([-3.0, 4.0])
        np.testing.assert_allclose(np.asarray(nb_abs), np.array([3.0, 4.0]))

    with ConfigContext(backend="numba", eager_mode=True):
        dense_backend = numba_kernels._resolve_dense_backend()
        assert dense_backend.backend_name == "pure_python"


def test_numpy_gather_index_numpy_hook() -> None:
    """Verify `gather_eager` unwraps index objects exposing a `.numpy()` method."""

    class _IndexWrapper(MockTestStub):
        """Index wrapper providing `.numpy()` conversion."""

        @staticmethod
        def numpy() -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
            """Return underlying NumPy integer index array.

            Returns:
                np.ndarray[tuple[int, ...], np.dtype[np.int64]]: Index array.
            """
            return np.array([1, 0], dtype=np.int64)

    src = np.array([10.0, 20.0], dtype=np.float32)
    out = gather_eager(np, src, 0, _IndexWrapper())
    np.testing.assert_allclose(np.asarray(out), np.array([20.0, 10.0], dtype=np.float32))
    out_plain = gather_eager(np, src, 0, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(np.asarray(out_plain), np.array([10.0, 20.0], dtype=np.float32))


def test_symbolic_collective_control_flow_upsample_and_joining_branches() -> None:
    """Verify symbolic `reduce_scatter`, `_map_fn_eager_get_length`, `upsample_nearest`, and `stack` branches."""
    assert _map_fn_eager_get_length(None) == 0

    class _NamedDType:
        """Custom dtype descriptor with `.name`."""

        def __init__(self, name: str) -> None:
            """Initialize named dtype descriptor.

            Args:
                name (str): Custom dtype name.
            """
            self.name = name

    class _NamedTensorHolder(MockTestStub):
        """Tensor holder exposing `_NamedDType`."""

        dtype = _NamedDType("INT16")

    class _DunderDesc:
        """Custom dtype descriptor instance exposing `.__name__`."""

        __name__ = "float16"

    class _DunderTensorHolder(MockTestStub):
        """Tensor holder exposing `_DunderDesc`."""

        dtype = _DunderDesc()

    assert _resolve_stack_dtype_str(_NamedTensorHolder()) == "int16"
    assert _resolve_stack_dtype_str(_DunderTensorHolder()) == "float16"
    assert _resolve_stack_dtype_str("plain_scalar") == "float32"

    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        tape.start_tracing("symbolic_fallbacks")
        try:
            sym_tensor = Tensor(
                ProxyTensor("sym_in", ("N", "C", "H", "W")),
                TensorConfig(shape=("N", "C", "H", "W"), dtype=DType.Float32, device="cpu"),
            )
            up_out = upsample_nearest(sym_tensor, scale_factor=2.0)
            assert isinstance(up_out, Tensor)

            sym_sc = Tensor(
                ProxyTensor("sym_sc", ("B", 4)),
                TensorConfig(shape=("B", 4), dtype=DType.Float32, device="cpu"),
            )
            sc_out = reduce_scatter(sym_sc, scatter_dim=0)
            assert isinstance(sc_out, Tensor)
            assert sc_out.shape == ("B", 4)

            class _UnknownDTypeTensor(MockTensorStub):
                """Symbolic tensor-like object with an unrecognized custom `.dtype`."""

                def __init__(self) -> None:
                    """Initialize stub attributes."""
                    super().__init__()
                    self.shape = (2,)
                    self.dtype = _NamedDType("unknown_custom_dtype")
                    self.data = ProxyTensor("s1", (2,))

            t2 = Tensor(ProxyTensor("s2", (2,)), TensorConfig((2,), DType.Float32, "cpu"))
            stacked = ensure_tensor(stack([_UnknownDTypeTensor(), t2], axis=0))
            assert stacked.dtype == DType.Float32
            assert stacked.shape == (2, 2)
        finally:
            traced_graph = tape.stop_tracing()

    assert traced_graph is not None
    op_types = {node.op_type for node in traced_graph.nodes.values()}
    assert "UpsampleNearest" in op_types
    assert "Stack" in op_types


def test_math_misc_op_tracing_emission() -> None:
    """Verify all 16 `math_misc` wrapper functions emit canonical IR operations during tracing."""
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        tape.start_tracing("math_misc_trace")
        try:
            vec = Tensor(ProxyTensor("vec", (4,)), TensorConfig((4,), DType.Float32, "cpu"))
            mat = Tensor(ProxyTensor("mat", (2, 2)), TensorConfig((2, 2), DType.Float32, "cpu"))

            math_misc.gradient(vec)
            math_misc.i0(vec)
            math_misc.interp(vec, vec, vec)
            math_misc.median(vec)
            math_misc.modf(vec)
            math_misc.piecewise(vec, [vec], [vec])
            math_misc.trapezoid(vec)
            math_misc.kron(mat, mat)
            math_misc.rot90(mat)
            math_misc.tri(3)
            math_misc.tril(mat)
            math_misc.trim_zeros(vec)
            math_misc.triu(mat)
            math_misc.vander(vec)
            math_misc.intersect1d(vec, vec)
            math_misc.unwrap(vec)
        finally:
            traced_graph = tape.stop_tracing()

    assert traced_graph is not None
    emitted_ops = {node.op_type for node in traced_graph.nodes.values()}
    expected_ops = {
        "Gradient",
        "I0",
        "Interp",
        "Median",
        "Modf",
        "Piecewise",
        "Trapezoid",
        "Kron",
        "Rot90",
        "Tri",
        "Tril",
        "TrimZeros",
        "Triu",
        "Vander",
        "Intersect1d",
        "Unwrap",
    }
    assert expected_ops.issubset(emitted_ops)
