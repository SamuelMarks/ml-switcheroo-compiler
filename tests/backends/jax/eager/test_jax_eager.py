# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_jax_eager."""

from __future__ import annotations

import types
import typing
from pathlib import Path

import pytest

from ml_switcheroo_compiler.backends.jax.eager import (
    _execute_accumulate_n,
    _execute_adaptive_avg_pool,
    _execute_bessel_jn,
    _execute_binom_cdf,
    _execute_binom_pmf,
    _execute_cast,
    _execute_convolve2d,
    _execute_ragged_tensor_to_dense,
    _execute_unsorted_segment_max,
    _execute_unsorted_segment_min,
    _execute_unsorted_segment_prod,
    _execute_unsorted_segment_sum,
    _execute_variance,
    _execute_welch,
    execute_op,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import (
    EagerTensorProtocol,
    MockTestStub,
    adapt_backend_module,
    extract_test_shape,
    require_callable,
)

try:
    import jax
except ImportError:
    pytest.skip("JAX not available", allow_module_level=True)
if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerCallableProtocol, EagerTensorProtocol


class _TestJaxEagerFunctionsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_jax_eager_functions` split helpers."""

    jnp: types.ModuleType
    res: EagerTensorProtocol | OpValue


def _test_jax_eager_functions_part_1(_state: _TestJaxEagerFunctionsState) -> None:
    """Execute part 1 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")
    res = _execute_adaptive_avg_pool(jnp.ones((2, 2)), 1)
    assert extract_test_shape(res) == (2, 1)
    res2 = _execute_adaptive_avg_pool(jnp.ones((2, 2)), (1, 1))
    assert extract_test_shape(res2) == (1, 1)
    res = _execute_accumulate_n([jnp.ones((2, 2)), jnp.ones((2, 2))])
    assert jnp.all(res == 2.0)
    res = _execute_binom_cdf(1, 2, 0.5)
    assert res is not None
    res = _execute_binom_pmf(1, 2, 0.5)
    assert res is not None
    res = _execute_convolve2d(jnp.ones((2, 2)), jnp.ones((2, 2)))
    assert res is not None
    res = _execute_welch(jnp.ones((8,)), nperseg=4)
    assert res is not None
    res = _execute_bessel_jn(1, 2.0)
    assert res is not None
    res = _execute_unsorted_segment_sum(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert ensure_ndarray(res)[0] == 3
    res = _execute_unsorted_segment_max(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert ensure_ndarray(res)[0] == 2
    res = _execute_unsorted_segment_min(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert ensure_ndarray(res)[0] == 1
    res = _execute_unsorted_segment_prod(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert ensure_ndarray(res)[0] == 2
    res = _execute_variance(jnp.array([1.0, 2.0, 3.0]))
    _state.update({"res": res})
    _state.update({"jnp": jnp})


def _test_jax_eager_functions_part_2(_state: _TestJaxEagerFunctionsState) -> None:
    """Execute part 2 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (jnp, res) = (_state["jnp"], _state["res"])
    assert res is not None

    class MockDtype(MockTestStub):
        """Test class MockDtype."""

        value = "int4"

    res = _execute_cast(jnp.array([1.0]), dtype=MockDtype())
    assert adapt_backend_module(res).require_symbol("dtype") == jnp.int8
    res = _execute_cast(jnp.array([1.0]), MockDtype())
    assert adapt_backend_module(res).require_symbol("dtype") == jnp.int8

    class MockDtype2(MockTestStub):
        """Test class MockDtype2."""

        value = "bfloat16"

    res = _execute_cast(jnp.array([1.0]), MockDtype2())
    assert adapt_backend_module(res).require_symbol("dtype") == jnp.bfloat16

    class MockDtype3(MockTestStub):
        """Test class MockDtype3."""

        value = "float16"

    res = _execute_cast(jnp.array([1.0]), MockDtype3())
    assert adapt_backend_module(res).require_symbol("dtype") == jnp.float16

    class MockDtype4(MockTestStub):
        """Test class MockDtype4."""

        value = "float8"

    res = _execute_cast(jnp.array([1.0]), MockDtype4())
    _state.update({"res": res})
    _state.update({"jnp": jnp})


def _test_jax_eager_functions_part_3(_state: _TestJaxEagerFunctionsState) -> None:
    """Execute part 3 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (jnp, res) = (_state["jnp"], _state["res"])
    assert res is not None

    class MockDtype5(MockTestStub):
        """Test class MockDtype5."""

        value = "float32"

    res = _execute_cast(jnp.array([1.0]), MockDtype5())
    assert adapt_backend_module(res).require_symbol("dtype") == jnp.float32
    res = _execute_ragged_tensor_to_dense([jnp.array([1.0]), jnp.array([1.0, 2.0])])
    assert adapt_backend_module(res).require_symbol("shape") == (2, 2)
    res = _execute_ragged_tensor_to_dense(jnp.array([1.0]))
    assert adapt_backend_module(res).require_symbol("shape") == (1,)
    res = execute_op(None, "Add", jnp.array([1.0]), jnp.array([2.0]))
    assert ensure_ndarray(res)[0] == 3.0
    res = execute_op(None, "Variance", jnp.array([1.0, 2.0, 3.0]))
    assert res is not None
    res = execute_op(None, "Cumprod", jnp.array([1.0, 2.0, 3.0]))
    assert res is not None
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    @global_eager_registry.register("DummyGlobalOp")
    def _dummy_global_op(jnp_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function _dummy_global_op.

        Args:
            jnp_module: Test parameter `jnp_module`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `_dummy_global_op`.
        """
        return adapt_backend_module(jnp_module).require_symbol("array")([42.0])

    res = execute_op(None, "DummyGlobalOp", jnp.array([1.0]))
    assert ensure_ndarray(res)[0] == 42.0
    _state.update({"res": res})
    _state.update({"jnp": jnp})


def _test_jax_eager_functions_part_4(_state: _TestJaxEagerFunctionsState) -> None:
    """Execute part 4 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError):
        execute_op(None, "UnknownOp123")


def test_jax_eager_functions() -> None:
    """Test function test_jax_eager_functions."""
    _state: _TestJaxEagerFunctionsState = {}
    _test_jax_eager_functions_part_1(_state)
    _test_jax_eager_functions_part_2(_state)
    _test_jax_eager_functions_part_3(_state)
    _test_jax_eager_functions_part_4(_state)
    assert len(_state) > 0


class _TestJaxEagerExecuteOpLambdasState(typing.TypedDict, total=False):
    """Typed shared state for `_test_jax_eager_execute_op_lambdas` split helpers."""

    _OP_DISPATCH: typing.Mapping[str, OpValue]
    jnp: types.ModuleType
    mock: types.ModuleType


def _test_jax_eager_execute_op_lambdas_part_1(_state: _TestJaxEagerExecuteOpLambdasState) -> None:
    """Execute part 1 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import _OP_DISPATCH

    _OP_DISPATCH["ActivityRegularization"](1.0)
    _OP_DISPATCH["AdaptiveMaxPool3D_Indices"](jnp.ones((2, 2)), 1)
    _OP_DISPATCH["AdaptiveLogSoftmaxWithLoss"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["Adjoint"](jnp.ones((2, 2)))
    from unittest import mock

    with (
        mock.patch("jax.lax.all_gather", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.all_to_all", lambda x, *args, **kw: x, create=True),
        mock.patch("jax.lax.psum", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.pmax", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.reduce_scatter", lambda x, *args, **kw: x, create=True),
    ):
        _OP_DISPATCH["AllGather"](jnp.ones((2,)))
        _OP_DISPATCH["AllToAll"](jnp.ones((2,)))
        _OP_DISPATCH["AllReduce"](jnp.ones((2,)), op_type="sum")
        _OP_DISPATCH["AllReduce"](jnp.ones((2,)), op_type="max")
        _OP_DISPATCH["ReduceScatter"](jnp.ones((2,)), op_type="sum")
        _OP_DISPATCH["ReduceScatter"](jnp.ones((2,)), op_type="max")
    _OP_DISPATCH["AlphaDropout"](jnp.ones((2,)))
    _OP_DISPATCH["AsString"](jnp.ones((2,)))
    _OP_DISPATCH["Assert"](True, jnp.ones((2,)))
    _OP_DISPATCH["Assign"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssignAdd"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssignSub"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssignVariable"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssociativeScan"](jnp.ones((2,)))
    _state.update({"_OP_DISPATCH": _OP_DISPATCH, "jnp": jnp, "mock": mock})


def _test_jax_eager_execute_op_lambdas_part_2(_state: _TestJaxEagerExecuteOpLambdasState) -> None:
    """Execute part 2 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    (_OP_DISPATCH, jnp) = (_state["_OP_DISPATCH"], _state["jnp"])
    require_callable(_OP_DISPATCH["AssociativeScan"])(lambda x: x, require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Frombuffer"])(b"\x00\x00\x00\x00", dtype=jnp.float32)
    require_callable(_OP_DISPATCH["Fft2"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Fftfreq"])(2)
    require_callable(_OP_DISPATCH["Fftn"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Fftnd"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Fftshift"])(require_callable(jnp.ones)((2, 2)))
    import jax.nn

    if adapt_backend_module(jax.nn).has_symbol("hard_silu"):
        require_callable(_OP_DISPATCH["HardSilu"])(require_callable(jnp.ones)((2,)))
    if adapt_backend_module(jax.nn).has_symbol("hard_swish"):
        require_callable(_OP_DISPATCH["HardSwish"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Hfft"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Ifft"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Ifft2"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Ifftn"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Ifftnd"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Ifftshift"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Ihfft"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Irfft"])(require_callable(jnp.ones)((2,)))
    _state.update({"_OP_DISPATCH": _OP_DISPATCH, "jnp": jnp})


def _test_jax_eager_execute_op_lambdas_part_3(_state: _TestJaxEagerExecuteOpLambdasState) -> None:
    """Execute part 3 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    (_OP_DISPATCH, jnp, mock) = (_state["_OP_DISPATCH"], _state["jnp"], _state["mock"])
    require_callable(_OP_DISPATCH["Irfft2"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Irfftn"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Irfftnd"])(require_callable(jnp.ones)((2, 2)))
    with mock.patch("jax.nn.mish", create=True, new=None), mock.patch("jax.nn.squareplus", create=True, new=None):
        import jax.nn

        if adapt_backend_module(jax.nn).has_symbol("mish"):
            del jax.nn.mish
        if adapt_backend_module(jax.nn).has_symbol("squareplus"):
            del jax.nn.squareplus
        require_callable(_OP_DISPATCH["Mish"])(require_callable(jnp.ones)((2,)))
        require_callable(_OP_DISPATCH["Squareplus"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["LogSoftmax"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Mish"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["OneHot"])(require_callable(jnp.array)([0, 1]), 2)
    require_callable(_OP_DISPATCH["OneHot"])(require_callable(jnp.array)([0, 1]), depth=2)
    require_callable(_OP_DISPATCH["Rfft"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Rfft2"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Rfftfreq"])(2)
    require_callable(_OP_DISPATCH["Rfftn"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Rfftnd"])(require_callable(jnp.ones)((2, 2)))
    require_callable(_OP_DISPATCH["Sigmoid"])(require_callable(jnp.ones)((2,)))
    require_callable(_OP_DISPATCH["Softmax"])(require_callable(jnp.ones)((2,)))
    _state.update({"_OP_DISPATCH": _OP_DISPATCH, "jnp": jnp, "mock": mock})


def _test_jax_eager_execute_op_lambdas_part_4(_state: _TestJaxEagerExecuteOpLambdasState) -> None:
    """Execute part 4 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    (_OP_DISPATCH, jnp) = (_state["_OP_DISPATCH"], _state["jnp"])
    require_callable(_OP_DISPATCH["Squareplus"])(require_callable(jnp.ones)((2,)))


def test_jax_eager_execute_op_lambdas() -> None:
    """Test function test_jax_eager_execute_op_lambdas."""
    _state: _TestJaxEagerExecuteOpLambdasState = {}
    _test_jax_eager_execute_op_lambdas_part_1(_state)
    _test_jax_eager_execute_op_lambdas_part_2(_state)
    _test_jax_eager_execute_op_lambdas_part_3(_state)
    _test_jax_eager_execute_op_lambdas_part_4(_state)
    assert len(_state) > 0


def test_jax_eager_execute_op_fallback() -> None:
    """Test function test_jax_eager_execute_op_fallback."""
    import unittest.mock as mock

    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import execute_op

    with mock.patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=None):
        with mock.patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            with pytest.raises(Exception, match=".*"):
                execute_op(None, "Mish", jnp.array([1.0]))
    res = execute_op(None, "Mul", jnp.array([1.0]), jnp.array([2.0]))
    assert ensure_ndarray(res)[0] == 2.0
    res = execute_op(None, "Sub", jnp.array([1.0]), jnp.array([2.0]))
    assert ensure_ndarray(res)[0] == -1.0
    res = execute_op(None, "Div", jnp.array([1.0]), jnp.array([2.0]))
    assert ensure_ndarray(res)[0] == 0.5
    res = execute_op(None, "Eig", jnp.eye(2))
    assert res is not None


def test_jax_eager_extra() -> None:
    """Test function test_jax_eager_extra."""
    import pytest

    from ml_switcheroo_compiler.backends.jax.eager import _execute_accumulate_n, _execute_adaptive_avg_pool

    assert _execute_adaptive_avg_pool([1, 2], 1) is not None
    with pytest.raises(ValueError):
        _execute_accumulate_n([])


def test_jax_adaptive_max_pool_tuple_size() -> None:
    """Test function test_jax_adaptive_max_pool_tuple_size."""
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import _execute_adaptive_max_pool

    operand = jnp.ones((1, 1, 4, 4))
    res = _execute_adaptive_max_pool(operand, (2, 2))
    assert extract_test_shape(res) == (1, 1, 2, 2)
