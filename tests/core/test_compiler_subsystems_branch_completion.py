"""Targeted branch-completion tests for core, distributed, export, grad, IR, ops, and serialization."""

from __future__ import annotations

import pathlib
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    CallableObjectProtocol,
    OpValue,
    VariadicCallableProtocol,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.distributed.host_collectives import pmean
from ml_switcheroo_compiler.export.aot import (
    _COMPILATION_CACHE,
    _compile_with_generator_instance,
    _try_static_compile_aot,
)
from ml_switcheroo_compiler.grad.custom_jvp_ops import _to_constant_attr
from ml_switcheroo_compiler.grad.utils import DifferentiableFn
from ml_switcheroo_compiler.ir.callback_registry import register_custom_rule_handle
from ml_switcheroo_compiler.ops.binary import _safe_get_op
from ml_switcheroo_compiler.ops.creation.api_basic import _unpack_shape
from ml_switcheroo_compiler.ops.normalization.basic import GroupMean, GroupVariance
from ml_switcheroo_compiler.ops.shape.joining import stack
from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from ml_switcheroo_compiler.serialization.utils import _extract_arr_bytes, _is_npz_files
from ml_switcheroo_compiler.tracing import global_tracing_state
from tests.testing_utils import MockTensorStub, extract_test_shape


class _InheritedCompileAotGenerator(BaseGenerator):
    """BaseGenerator subclass leaving `compile_aot` un-overridden."""

    def generate(self, graph: LogicalGraph | None = None) -> str:
        """Emit trivial source code.

        Args:
            graph (LogicalGraph | None): Optional logical graph.

        Returns:
            str: Trivial Python code string.
        """
        del graph
        return """def forward(x):
    return x
"""


class _CustomInstanceAotGenerator:
    """Generator stub whose instance `compile_aot` returns a concrete callable."""

    def __init__(self, graph: LogicalGraph) -> None:
        """Store `graph` for instance AOT compilation.

        Args:
            graph (LogicalGraph): Target logical graph.
        """
        self.graph = graph

    def compile_aot(
        self,
        graph: LogicalGraph,
        **kwargs: OpValue,
    ) -> VariadicCallableProtocol:
        """Return a compiled doubling function for AOT instance execution.

        Args:
            graph (LogicalGraph): Target logical graph.
            **kwargs (OpValue): Compilation options.

        Returns:
            VariadicCallableProtocol: Callable doubling the first integer argument.
        """
        del graph, kwargs

        def _compiled(*call_args: OpValue, **call_kw: OpValue) -> OpValue:
            """Return twice the first numeric argument.

            Args:
                *call_args (OpValue): Positional call arguments.
                **call_kw (OpValue): Keyword call arguments.

            Returns:
                OpValue: Doubled numeric result.
            """
            del call_kw
            first = call_args[0]
            assert isinstance(first, int)
            return first * 2

        return _compiled


class _BackendWithArrayOnly:
    """Minimal backend stub lacking `numpy` and `asarray` while providing `array`."""

    def array(self, data: OpValue) -> list[int]:
        """Wrap `data` into a tagged integer list.

        Args:
            data (OpValue): Raw tensor payload.

        Returns:
            list[int]: Converted list marker.
        """
        assert isinstance(data, int)
        return [data, 99]


class _BackendWithNoConverters:
    """Minimal backend stub lacking `numpy`, `asarray`, `array`, and `item`."""


class _ScalarItemHolder:
    """Scalar wrapper implementing `.item()` without backend `item` support."""

    def item(self) -> float:
        """Extract scalar float value.

        Returns:
            float: Constant scalar `7.5`.
        """
        return 7.5


class _NamedNonCallable:
    """Object satisfying `CallableObjectProtocol` (`__name__`) without implementing `__call__`."""

    @property
    def __name__(self) -> str:
        """Return symbol name.

        Returns:
            str: Mock function identifier.
        """
        return "non_callable_named_obj"


class _BrokenIntDim(int):
    """`int` subclass whose `__int__` raises `ValueError` to exercise `_unpack_shape` fallback."""

    def __int__(self) -> int:
        """Raise `ValueError` during integer coercion.

        Returns:
            int: Never returns normally.

        Raises:
            ValueError: Always raised to test symbolic fallback.
        """
        raise ValueError("symbolic dim cannot coerce to int")


class _NonByteDataHolder:
    """Object exposing `.data` without `.tobytes()`."""

    def __init__(self) -> None:
        """Initialize `.data` with a non-byte list payload."""
        self.data: OpValue = [1, 2, 3]


class _FakeNpzArchive:
    """Archive stub exposing `.shape`, `.files`, and `__getitem__` for `_is_npz_files`."""

    def __init__(self) -> None:
        """Initialize `.shape` and `.files` list."""
        self.shape: tuple[int, ...] = ()
        self.files: list[str] = ["arr_0"]

    def __getitem__(self, key: str) -> list[float]:
        """Return dummy float list for `key`.

        Args:
            key (str): Archive member key.

        Returns:
            list[float]: Dummy float array data.
        """
        del key
        return [1.0]


def test_device_equality_variants() -> None:
    """Verify `Device.__eq__` against `DeviceType`, colon-indexed strings, and unrelated types."""
    cpu0 = Device(DeviceType.CPU, 0)
    gpu1 = Device(DeviceType.GPU, 1)

    assert (cpu0 == DeviceType.CPU) is True
    assert (gpu1 == DeviceType.GPU) is False
    assert (gpu1 == "gpu:1") is True
    assert (gpu1 == "gpu:0") is False
    assert (cpu0 == "cpu") is True
    assert (cpu0 == 42) is False


def test_tensor_mixins_numpy_and_item_fallbacks() -> None:
    """Verify `Tensor.numpy()` and `Tensor.item()` fallback chains when backend methods are absent."""
    t_int = Tensor(5, TensorConfig(shape=(), dtype=DType.Int32, device="cpu"))
    with patch(
        "ml_switcheroo_compiler.backends.registry.get_active_backend",
        return_value=_BackendWithArrayOnly(),
    ):
        assert t_int.numpy() == [5, 99]

    with patch(
        "ml_switcheroo_compiler.backends.registry.get_active_backend",
        return_value=_BackendWithNoConverters(),
    ):
        assert t_int.numpy() == 5

        t_item = Tensor(_ScalarItemHolder(), TensorConfig(shape=(), dtype=DType.Float32, device="cpu"))
        with patch.object(Tensor, "eval", return_value=t_item):
            assert t_item.item() == 7.5


def test_host_collectives_pmean_raw_array() -> None:
    """Verify `pmean` divides raw non-`Tensor` arrays via `_exec_collective_op`."""
    raw_arr = [4.0, 8.0]
    res = pmean(raw_arr, mesh_axis="data", all_ranks_data=[[2.0, 4.0], [6.0, 12.0]])
    np.testing.assert_allclose(np.asarray(res), np.array([4.0, 8.0]))


def test_aot_class_and_instance_compile_aot_branches() -> None:
    """Verify `_try_static_compile_aot` un-overridden/non-callable branches and instance `compile_aot`."""
    graph = LogicalGraph(name="aot_branch_test")
    assert _try_static_compile_aot(_InheritedCompileAotGenerator, graph, {}) is None

    class _NullCompileAotProvider:
        """Generator provider whose `compile_aot` returns `None` instead of a callable."""

        shape: tuple[int, ...] = ()

        @classmethod
        def compile_aot(cls, g: LogicalGraph, **kw: OpValue) -> None:
            """Return `None` to exercise non-callable `compiled_fn` branch.

            Args:
                g (LogicalGraph): Target graph.
                **kw (OpValue): Compilation options.
            """
            del g, kw

    assert _try_static_compile_aot(_NullCompileAotProvider, graph, {}) is None

    cache_key = "test_instance_compile_aot_key"
    _COMPILATION_CACHE.pop(cache_key, None)
    try:
        out = _compile_with_generator_instance(
            _CustomInstanceAotGenerator,
            graph,
            "custom_test_backend",
            cache_key,
            {},
            lambda x: x,
            (21,),
            {},
        )
        assert out == 42
        assert cache_key in _COMPILATION_CACHE
    finally:
        _COMPILATION_CACHE.pop(cache_key, None)


def test_custom_jvp_constant_attr_and_callback_registry_error() -> None:
    """Verify `_to_constant_attr` string fallback and `register_custom_rule_handle` `TypeError`."""
    dev = Device(DeviceType.CPU, 0)
    assert _to_constant_attr(dev) == str(dev)

    bad_fn: CallableObjectProtocol = _NamedNonCallable()
    typed_bad_fn: DifferentiableFn = bad_fn
    with pytest.raises(TypeError, match="Expected callable custom rule"):
        register_custom_rule_handle(typed_bad_fn)


def test_binary_safe_get_op_and_creation_unpack_shape_fallback() -> None:
    """Verify `_safe_get_op` unregistered fallback and `_unpack_shape` `ValueError` handling."""
    fallback_op = _safe_get_op("NonExistentBinaryOpXYZ")
    assert fallback_op.op_type == "NonExistentBinaryOpXYZ"

    bad_dim = _BrokenIntDim(3)
    unpacked = _unpack_shape((2, bad_dim))
    assert unpacked[0] == 2
    assert unpacked[1] is bad_dim


def test_normalization_group_shape_helpers_and_joining_stack_invalid_dtype() -> None:
    """Verify `GroupMean`/`GroupVariance`/`GroupNorm` shape inference edge cases and `stack` dtype fallback."""
    assert GroupMean().infer_shape((2, 8, 4), num_groups=4.0) == (2, 4, 1)
    assert GroupVariance().infer_shape((2, 8, 4), num_groups=True) == (2, 1, 1)
    assert GroupMean().infer_shape(None) == ()

    stub = MockTensorStub()
    stub.shape = (3,)
    pytest.MonkeyPatch().setattr(stub, "_dtype", "unknown_custom_dtype_xyz")
    global_tracing_state.start_tracing("stack_invalid_dtype_test")
    try:
        with ConfigContext(eager_mode=False):
            stacked = stack([stub, stub], axis=0)
        assert extract_test_shape(stacked) == (2, 3)
        assert isinstance(stacked, Tensor)
        assert stacked.dtype == DType.Float32
    finally:
        global_tracing_state.stop_tracing()


def test_serialization_pickle_error_and_utils_helpers(tmp_path: pathlib.Path) -> None:
    """Verify `PickleWeightFormat.load` non-dict `TypeError` and serialization utility branches.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture.
    """
    bad_pkl = tmp_path / "not_a_dict.pkl"
    with open(bad_pkl, "wb") as f:
        pickle.dump([1, 2, 3], f)

    with pytest.raises(TypeError, match="Expected dict weight payload"):
        PickleWeightFormat().load(str(bad_pkl))

    assert _extract_arr_bytes(_NonByteDataHolder()) == b""
    assert _is_npz_files(_FakeNpzArchive()) is True
    assert _is_npz_files([1, 2]) is False
