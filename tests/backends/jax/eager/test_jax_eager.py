# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_jax_eager."""

from __future__ import annotations

import typing

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
from tests.testing_utils import as_callable, as_tensor

try:
    import jax
except ImportError:
    pytest.skip("JAX not available", allow_module_level=True)
if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _test_jax_eager_functions_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")
    res = _execute_adaptive_avg_pool(jnp.ones((2, 2)), 1)
    assert as_tensor(getattr(res, "shape")) == (2, 1)
    res2 = _execute_adaptive_avg_pool(jnp.ones((2, 2)), (1, 1))
    assert as_tensor(getattr(res2, "shape")) == (1, 1)
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
    assert as_tensor(getattr(res, "__getitem__")(0)) == 3
    res = _execute_unsorted_segment_max(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert as_tensor(getattr(res, "__getitem__")(0)) == 2
    res = _execute_unsorted_segment_min(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert as_tensor(getattr(res, "__getitem__")(0)) == 1
    res = _execute_unsorted_segment_prod(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert as_tensor(getattr(res, "__getitem__")(0)) == 2
    res = _execute_variance(jnp.array([1.0, 2.0, 3.0]))
    getattr(_state, "update")({"res": res, "res2": res2})
    _state.update(locals())


def _test_jax_eager_functions_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (jnp, res) = (_state["jnp"], _state["res"])
    assert res is not None

    class MockDtype:
        """Test class MockDtype."""

        value = "int4"

    res = as_callable(_execute_cast)(as_callable(jnp.array)([1.0]), dtype=as_callable(MockDtype)())
    assert res.dtype == jnp.int8
    res = as_callable(_execute_cast)(as_callable(jnp.array)([1.0]), as_callable(MockDtype)())
    assert res.dtype == jnp.int8

    class MockDtype2:
        """Test class MockDtype2."""

        value = "bfloat16"

    res = as_callable(_execute_cast)(as_callable(jnp.array)([1.0]), as_callable(MockDtype2)())
    assert res.dtype == jnp.bfloat16

    class MockDtype3:
        """Test class MockDtype3."""

        value = "float16"

    res = as_callable(_execute_cast)(as_callable(jnp.array)([1.0]), as_callable(MockDtype3)())
    assert res.dtype == jnp.float16

    class MockDtype4:
        """Test class MockDtype4."""

        value = "float8"

    res = as_callable(_execute_cast)(as_callable(jnp.array)([1.0]), as_callable(MockDtype4)())
    getattr(_state, "update")({"MockDtype": MockDtype, "MockDtype2": MockDtype2, "MockDtype3": MockDtype3, "MockDtype4": MockDtype4})
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def _test_jax_eager_functions_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    (jnp, res) = (_state["jnp"], _state["res"])
    assert res is not None

    class MockDtype5:
        """Test class MockDtype5."""

        value = "float32"

    res = as_callable(_execute_cast)(as_callable(jnp.array)([1.0]), as_callable(MockDtype5)())
    assert res.dtype == jnp.float32
    res = as_tensor(_execute_ragged_tensor_to_dense([jnp.array([1.0]), jnp.array([1.0, 2.0])]))
    assert res.shape == (2, 2)
    res = as_tensor(_execute_ragged_tensor_to_dense(jnp.array([1.0])))
    assert res.shape == (1,)
    res = as_callable(execute_op)(None, "Add", as_callable(jnp.array)([1.0]), as_callable(jnp.array)([2.0]))
    assert res[0] == 3.0
    res = as_callable(execute_op)(None, "Variance", as_callable(jnp.array)([1.0, 2.0, 3.0]))
    assert res is not None
    res = as_callable(execute_op)(None, "Cumprod", as_callable(jnp.array)([1.0, 2.0, 3.0]))
    assert res is not None
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    @global_eager_registry.register("DummyGlobalOp")
    def _dummy_global_op(jnp_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function _dummy_global_op.

        Args:
            jnp_module: Test parameter `jnp_module`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        return as_tensor(getattr(jnp_module, "array"))([42.0])

    res = as_callable(execute_op)(None, "DummyGlobalOp", as_callable(jnp.array)([1.0]))
    assert res[0] == 42.0
    getattr(_state, "update")({"MockDtype5": MockDtype5, "_dummy_global_op": _dummy_global_op, "global_eager_registry": global_eager_registry, "res": res})
    _state.update(locals())


def _test_jax_eager_functions_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_jax_eager_functions.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError):
        as_callable(execute_op)(None, "UnknownOp123")
    getattr(_state, "update")({"BackendNotSupportedError": BackendNotSupportedError})


def test_jax_eager_functions() -> None:
    """Test function test_jax_eager_functions."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_jax_eager_functions_part_1(_state)
    _test_jax_eager_functions_part_2(_state)
    _test_jax_eager_functions_part_3(_state)
    _test_jax_eager_functions_part_4(_state)
    assert len(_state) > 0


def _test_jax_eager_execute_op_lambdas_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import _OP_DISPATCH

    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("ActivityRegularization")))(1.0)
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AdaptiveMaxPool3D_Indices")))(as_callable(jnp.ones)((2, 2)), 1)
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AdaptiveLogSoftmaxWithLoss")))(as_callable(jnp.ones)((2,)), as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("Adjoint")))(as_callable(jnp.ones)((2, 2)))
    from unittest import mock

    with (
        mock.patch("jax.lax.all_gather", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.all_to_all", lambda x, *args, **kw: x, create=True),
        mock.patch("jax.lax.psum", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.pmax", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.reduce_scatter", lambda x, *args, **kw: x, create=True),
    ):
        as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AllGather")))(as_callable(jnp.ones)((2,)))
        as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AllToAll")))(as_callable(jnp.ones)((2,)))
        as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AllReduce")))(as_callable(jnp.ones)((2,)), op_type="sum")
        as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AllReduce")))(as_callable(jnp.ones)((2,)), op_type="max")
        as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("ReduceScatter")))(as_callable(jnp.ones)((2,)), op_type="sum")
        as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("ReduceScatter")))(as_callable(jnp.ones)((2,)), op_type="max")
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AlphaDropout")))(as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AsString")))(as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("Assert")))(True, as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("Assign")))(as_callable(jnp.ones)((2,)), as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AssignAdd")))(as_callable(jnp.ones)((2,)), as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AssignSub")))(as_callable(jnp.ones)((2,)), as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AssignVariable")))(as_callable(jnp.ones)((2,)), as_callable(jnp.ones)((2,)))
    as_callable(as_tensor(getattr(_OP_DISPATCH, "__getitem__")("AssociativeScan")))(as_callable(jnp.ones)((2,)))
    getattr(_state, "update")({"_OP_DISPATCH": _OP_DISPATCH, "jnp": jnp, "mock": mock})
    _state.update(locals())


def _test_jax_eager_execute_op_lambdas_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    (_OP_DISPATCH, jnp) = (_state["_OP_DISPATCH"], _state["jnp"])
    _OP_DISPATCH["AssociativeScan"](lambda x: x, jnp.ones((2,)))
    _OP_DISPATCH["Frombuffer"](b"\x00\x00\x00\x00", dtype=jnp.float32)
    _OP_DISPATCH["Fft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Fftfreq"](2)
    _OP_DISPATCH["Fftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Fftnd"](jnp.ones((2, 2)))
    _OP_DISPATCH["Fftshift"](jnp.ones((2, 2)))
    import jax.nn

    if hasattr(jax.nn, "hard_silu"):
        _OP_DISPATCH["HardSilu"](jnp.ones((2,)))
    if hasattr(jax.nn, "hard_swish"):
        _OP_DISPATCH["HardSwish"](jnp.ones((2,)))
    _OP_DISPATCH["Hfft"](jnp.ones((2,)))
    _OP_DISPATCH["Ifft"](jnp.ones((2,)))
    _OP_DISPATCH["Ifft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ifftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ifftnd"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ifftshift"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ihfft"](jnp.ones((2,)))
    _OP_DISPATCH["Irfft"](jnp.ones((2,)))
    getattr(_state, "update")({"jax": jax})
    _state.update(locals())


def _test_jax_eager_execute_op_lambdas_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    (_OP_DISPATCH, jnp, mock) = (_state["_OP_DISPATCH"], _state["jnp"], _state["mock"])
    _OP_DISPATCH["Irfft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Irfftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Irfftnd"](jnp.ones((2, 2)))
    with mock.patch("jax.nn.mish", create=True, new=None), mock.patch("jax.nn.squareplus", create=True, new=None):
        import jax.nn

        if hasattr(jax.nn, "mish"):
            del jax.nn.mish
        if hasattr(jax.nn, "squareplus"):
            del jax.nn.squareplus
        _OP_DISPATCH["Mish"](jnp.ones((2,)))
        _OP_DISPATCH["Squareplus"](jnp.ones((2,)))
    _OP_DISPATCH["LogSoftmax"](jnp.ones((2,)))
    _OP_DISPATCH["Mish"](jnp.ones((2,)))
    _OP_DISPATCH["OneHot"](jnp.array([0, 1]), 2)
    _OP_DISPATCH["OneHot"](jnp.array([0, 1]), depth=2)
    _OP_DISPATCH["Rfft"](jnp.ones((2,)))
    _OP_DISPATCH["Rfft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Rfftfreq"](2)
    _OP_DISPATCH["Rfftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Rfftnd"](jnp.ones((2, 2)))
    _OP_DISPATCH["Sigmoid"](jnp.ones((2,)))
    _OP_DISPATCH["Softmax"](jnp.ones((2,)))
    getattr(_state, "update")({"jax": jax})
    _state.update(locals())


def _test_jax_eager_execute_op_lambdas_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_jax_eager_execute_op_lambdas.

    Args:
        _state: Shared test state dictionary.
    """
    (_OP_DISPATCH, jnp) = (_state["_OP_DISPATCH"], _state["jnp"])
    _OP_DISPATCH["Squareplus"](jnp.ones((2,)))


def test_jax_eager_execute_op_lambdas() -> None:
    """Test function test_jax_eager_execute_op_lambdas."""
    _state: dict[str, EagerTensorProtocol] = {}
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
                as_callable(execute_op)(None, "Mish", as_callable(jnp.array)([1.0]))
    res = as_callable(execute_op)(None, "Mul", as_callable(jnp.array)([1.0]), as_callable(jnp.array)([2.0]))
    assert as_tensor(getattr(res, "__getitem__")(0)) == 2.0
    res = as_callable(execute_op)(None, "Sub", as_callable(jnp.array)([1.0]), as_callable(jnp.array)([2.0]))
    assert as_tensor(getattr(res, "__getitem__")(0)) == -1.0
    res = as_callable(execute_op)(None, "Div", as_callable(jnp.array)([1.0]), as_callable(jnp.array)([2.0]))
    assert as_tensor(getattr(res, "__getitem__")(0)) == 0.5
    res = as_callable(execute_op)(None, "Eig", as_callable(jnp.eye)(2))
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
    assert as_tensor(getattr(res, "shape")) == (1, 1, 2, 2)
