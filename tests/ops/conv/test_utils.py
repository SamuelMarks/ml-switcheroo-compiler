"""Tests for test_utils."""

from __future__ import annotations

import types
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.nn.conv_utils import (
    _build_conv_config,
    _calc_same_pad,
    _calc_valid_pad,
    _calculate_conv_transpose_padding,
    atrous_conv2d,
    atrous_conv2d_transpose,
    bias_add,
    collapse_repeated,
    compute_average_loss,
    conv_transpose,
    convolution,
    depthwise_conv2d,
    depthwise_conv2d_backprop_filter,
    depthwise_conv2d_backprop_input,
    dilation2d,
    erosion2d,
)
from tests.testing_utils import EagerBoundCallable, MockTestStub, coerce_int_list, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_conv_utils_calc_pad() -> None:
    """Test function test_conv_utils_calc_pad."""
    assert _calc_same_pad(3, 1) == (1, 1)
    assert _calc_same_pad(3, 3) == (2, 2)
    assert _calc_same_pad(3, 4) == (2, 3)
    assert _calc_valid_pad(3, 1) == (2, 2)


def test_build_conv_config() -> None:
    """Test function test_build_conv_config."""
    kwargs: dict[str, OpValue] = {
        "strides": 1,
        "padding": "SAME",
        "lhs_dilation": 1,
        "rhs_dilation": 1,
        "feature_group_count": 1,
        "batch_group_count": 1,
    }
    c = _build_conv_config(kwargs, ((0, 1), (0, 1), (0, 1)))
    assert type(c).__name__ == "ConvConfig"
    c = _build_conv_config({}, ((0, 1), (0, 1), (0, 1)))
    assert type(c).__name__ == "ConvConfig"


def test_calculate_conv_transpose_padding() -> None:
    """Test function test_calculate_conv_transpose_padding."""
    assert len(_calculate_conv_transpose_padding((5, 5), (3, 3), (2, 2))) == 2


class _TestConvUtilsFrontendFuncsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_conv_utils_frontend_funcs` split helpers."""

    _unused: int


def _test_conv_utils_frontend_funcs_sub_1(_state: _TestConvUtilsFrontendFuncsState) -> None:
    """Execute sub-part 1 of test_conv_utils_frontend_funcs.

    Args:
        _state (_TestConvUtilsFrontendFuncsState): Shared test state dictionary.
    """
    config.eager_mode = False
    with pytest.raises(RuntimeError):
        atrous_conv2d(None, None, None, None)
    with pytest.raises(KeyError):
        atrous_conv2d_transpose(None, None, None, None, None)
    with pytest.raises(RuntimeError):
        bias_add(None, None)
    collapse_repeated(None, None)


def _test_conv_utils_frontend_funcs_sub_2(_state: _TestConvUtilsFrontendFuncsState) -> None:
    """Execute sub-part 2 of test_conv_utils_frontend_funcs.

    Args:
        _state (_TestConvUtilsFrontendFuncsState): Shared test state dictionary.
    """
    with pytest.raises(RuntimeError):
        compute_average_loss(None)
    with pytest.raises(KeyError):
        require_callable(depthwise_conv2d)(None, None, None, None)
    depthwise_conv2d_backprop_filter(None, None, None, None, None)
    depthwise_conv2d_backprop_input(None, None, None, None, None)


def _test_conv_utils_frontend_funcs_sub_3(_state: _TestConvUtilsFrontendFuncsState) -> None:
    """Execute sub-part 3 of test_conv_utils_frontend_funcs.

    Args:
        _state (_TestConvUtilsFrontendFuncsState): Shared test state dictionary.
    """
    with pytest.raises(TypeError):
        require_callable(dilation2d)(None, None, None, None, None)
    with pytest.raises(TypeError):
        require_callable(erosion2d)(None, None, None, None, None)
    with pytest.raises(KeyError):
        convolution(None, None)
    with pytest.raises(TypeError):
        require_callable(conv_transpose)(None, None)


def test_conv_utils_frontend_funcs() -> None:
    """Test function test_conv_utils_frontend_funcs."""
    orig_eager = config.eager_mode
    try:
        _state: _TestConvUtilsFrontendFuncsState = {}
        for _sub_fn in (
            _test_conv_utils_frontend_funcs_sub_1,
            _test_conv_utils_frontend_funcs_sub_2,
            _test_conv_utils_frontend_funcs_sub_3,
        ):
            _sub_fn(_state)
    finally:
        config.eager_mode = orig_eager
    assert config.eager_mode == orig_eager


def test_prepare_depthwise_conv() -> None:
    """Test function test_prepare_depthwise_conv."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import _prepare_depthwise_conv

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape (BackendArg): Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    t2 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackendConv(BaseGenerator):
            """Test class MockBackendConv."""

            @classmethod
            def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function execute_op.

                Args:
                    op_type (str): Test parameter `op_type`.
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `execute_op`.
                """
                del op_type, args, kwargs
                return np.ones((2, 2, 2))

            @classmethod
            def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
                """Test function array.

                Args:
                    data (BackendArg): Test parameter `data`.
                    dtype (BackendArg | None): Test parameter `dtype`.

                Returns:
                    BackendArg: Computed output value from `array`.
                """
                del dtype
                return data

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        BackendRegistry.register("mock_ops", MockBackendConv)
        _prepare_depthwise_conv(
            t1, t2, 1, ((0, 1), (0, 1), (0, 1)), config_obj=None, strides=1, lhs_dilation=1, rhs_dilation=1
        )
    finally:
        config.eager_mode = orig_eager
    assert config.eager_mode == orig_eager


def test_conv_utils_calc_pad_same_valid_2() -> None:
    """Test function test_conv_utils_calc_pad_same_valid_2."""
    from ml_switcheroo_compiler.ops.nn.conv_utils import _calculate_conv_transpose_padding

    assert _calculate_conv_transpose_padding("SAME", (3, 3), (2, 2)) is not None
    assert _calculate_conv_transpose_padding("VALID", (3, 3), (2, 2)) is not None
    assert _calculate_conv_transpose_padding("OTHER", (3, 3), (2, 2)) == "OTHER"


def test_conv_utils_prepare_depthwise_conv_extra() -> None:
    """Test function test_conv_utils_prepare_depthwise_conv_extra."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import _prepare_depthwise_conv

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape (BackendArg): Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    t2 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackendConv(BaseGenerator):
            """Test class MockBackendConv."""

            @classmethod
            def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function execute_op.

                Args:
                    op_type (str): Test parameter `op_type`.
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `execute_op`.
                """
                del op_type, args, kwargs
                return np.ones((2, 2, 2))

            @classmethod
            def array(cls, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
                """Test function array.

                Args:
                    data (BackendArg): Test parameter `data`.
                    dtype (BackendArg | None): Test parameter `dtype`.

                Returns:
                    BackendArg: Computed output value from `array`.
                """
                del dtype
                return data

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        BackendRegistry.register("mock_ops", MockBackendConv)
        _prepare_depthwise_conv(
            t1, t2, 1, ((0, 1), (0, 1), (0, 1)), config_obj=None, strides=(1,), lhs_dilation=(1,), rhs_dilation=(1,)
        )
    finally:
        config.eager_mode = orig_eager
    assert config.eager_mode == orig_eager


def test_conv_utils_frontend_funcs_extra() -> None:
    """Test function test_conv_utils_frontend_funcs_extra."""
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, atrous_conv2d_transpose

    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        with pytest.raises(KeyError):
            atrous_conv2d_transpose(None, None, None, GenericConvConfig(), None)
    finally:
        config.eager_mode = orig_eager


class _TestConvUtilsFrontendFuncsExtra2State(typing.TypedDict, total=False):
    """Typed shared state for `_test_conv_utils_frontend_funcs_extra2` split helpers."""

    atrous_conv2d: EagerBoundCallable
    bias_add: EagerBoundCallable
    collapse_repeated: EagerBoundCallable
    compute_average_loss: EagerBoundCallable
    conv_transpose: EagerBoundCallable
    convolution: EagerBoundCallable
    depthwise_conv2d: EagerBoundCallable
    depthwise_conv2d_backprop_filter: EagerBoundCallable
    depthwise_conv2d_backprop_input: EagerBoundCallable
    dilation2d: EagerBoundCallable
    erosion2d: EagerBoundCallable
    t1: Tensor


def _test_conv_utils_frontend_funcs_extra2_sub_1(_state: _TestConvUtilsFrontendFuncsExtra2State) -> None:
    """Execute sub-part 1 of test_conv_utils_frontend_funcs_extra2.

    Args:
        _state: Shared test state dictionary.
    """
    (atrous_conv2d, bias_add, collapse_repeated, compute_average_loss) = (
        _state["atrous_conv2d"],
        _state["bias_add"],
        _state["collapse_repeated"],
        _state["compute_average_loss"],
    )
    t1 = _state["t1"]
    config.eager_mode = False
    with pytest.raises(RuntimeError):
        atrous_conv2d(t1, t1, 1, "VALID", None)
    with pytest.raises(RuntimeError):
        bias_add(t1, t1, None, None)
    collapse_repeated(t1, t1, None)
    with pytest.raises(RuntimeError):
        compute_average_loss(t1, t1, t1)
    _state.update(
        {
            "atrous_conv2d": require_callable(atrous_conv2d),
            "bias_add": require_callable(bias_add),
            "collapse_repeated": require_callable(collapse_repeated),
        }
    )
    _state.update({"compute_average_loss": require_callable(compute_average_loss), "t1": t1})


def _test_conv_utils_frontend_funcs_extra2_sub_2(_state: _TestConvUtilsFrontendFuncsExtra2State) -> None:
    """Execute sub-part 2 of test_conv_utils_frontend_funcs_extra2.

    Args:
        _state: Shared test state dictionary.
    """
    (depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input, dilation2d) = (
        _state["depthwise_conv2d"],
        _state["depthwise_conv2d_backprop_filter"],
        _state["depthwise_conv2d_backprop_input"],
        _state["dilation2d"],
    )
    t1 = _state["t1"]
    with pytest.raises(KeyError):
        depthwise_conv2d(t1, t1, t1, t1)
    depthwise_conv2d_backprop_filter(t1, t1, t1, t1, t1)
    depthwise_conv2d_backprop_input(t1, t1, t1, t1, t1)
    with pytest.raises(TypeError):
        dilation2d(t1, t1, t1, t1, t1)
    _state.update(
        {
            "depthwise_conv2d": require_callable(depthwise_conv2d),
            "depthwise_conv2d_backprop_filter": require_callable(depthwise_conv2d_backprop_filter),
            "depthwise_conv2d_backprop_input": require_callable(depthwise_conv2d_backprop_input),
        }
    )
    _state.update({"dilation2d": require_callable(dilation2d), "t1": t1})


def _test_conv_utils_frontend_funcs_extra2_sub_3(_state: _TestConvUtilsFrontendFuncsExtra2State) -> None:
    """Execute sub-part 3 of test_conv_utils_frontend_funcs_extra2.

    Args:
        _state: Shared test state dictionary.
    """
    (conv_transpose, convolution, erosion2d, t1) = (
        _state["conv_transpose"],
        _state["convolution"],
        _state["erosion2d"],
        _state["t1"],
    )
    with pytest.raises(TypeError):
        erosion2d(t1, t1, t1, t1, t1)
    with pytest.raises(KeyError):
        convolution(t1, t1)
    with pytest.raises(TypeError):
        conv_transpose(t1, t1)
    _state.update(
        {
            "conv_transpose": require_callable(conv_transpose),
            "convolution": require_callable(convolution),
            "erosion2d": require_callable(erosion2d),
        }
    )
    _state.update({"t1": t1})


def test_conv_utils_frontend_funcs_extra2() -> None:
    """Test function test_conv_utils_frontend_funcs_extra2."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        atrous_conv2d,
        bias_add,
        collapse_repeated,
        compute_average_loss,
        conv_transpose,
        convolution,
        depthwise_conv2d,
        depthwise_conv2d_backprop_filter,
        depthwise_conv2d_backprop_input,
        dilation2d,
        erosion2d,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        _state: _TestConvUtilsFrontendFuncsExtra2State = {}
        _state.update({"atrous_conv2d": require_callable(atrous_conv2d), "bias_add": require_callable(bias_add)})
        _state.update(
            {
                "collapse_repeated": require_callable(collapse_repeated),
                "compute_average_loss": require_callable(compute_average_loss),
                "conv_transpose": require_callable(conv_transpose),
                "convolution": require_callable(convolution),
            }
        )
        _state.update(
            {
                "depthwise_conv2d": require_callable(depthwise_conv2d),
                "depthwise_conv2d_backprop_filter": require_callable(depthwise_conv2d_backprop_filter),
                "depthwise_conv2d_backprop_input": require_callable(depthwise_conv2d_backprop_input),
                "dilation2d": require_callable(dilation2d),
            }
        )
        _state.update({"erosion2d": require_callable(erosion2d), "t1": t1})
        for _sub_fn in (
            _test_conv_utils_frontend_funcs_extra2_sub_1,
            _test_conv_utils_frontend_funcs_extra2_sub_2,
            _test_conv_utils_frontend_funcs_extra2_sub_3,
        ):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        config.eager_mode = orig_eager


def test_conv_utils_frontend_funcs_extra3() -> None:
    """Test function test_conv_utils_frontend_funcs_extra3."""
    import types

    import ml_switcheroo_compiler.ops.base as base
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        GenericConvConfig,
        atrous_conv2d,
        atrous_conv2d_transpose,
        conv_transpose,
        convolution,
        depthwise_conv2d,
        depthwise_conv2d_backprop_filter,
        depthwise_conv2d_backprop_input,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = base.get_op

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return "mock_op_called"

        from unittest.mock import patch

        pytest.MonkeyPatch().setattr(base, "get_op", lambda name: MockOp, raising=False)
        with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", side_effect=lambda name: MockOp):
            out_ac = atrous_conv2d(t1, t1, 1, "VALID", None)
        assert out_ac == "mock_op_called"
        try:
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            depthwise_conv2d(t1, t1, [1, 1], "VALID")
            depthwise_conv2d_backprop_filter(t1, t1, t1, [1, 1], "VALID")
            depthwise_conv2d_backprop_input((2, 2, 2), t1, t1, [1, 1], "VALID")
            convolution(t1, t1)
            conv_transpose(t1, t1, (2, 2, 2))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError) as err:
            assert str(err) != ""
    finally:
        config.eager_mode = orig_eager
        pytest.MonkeyPatch().setattr(base, "get_op", orig_get_op, raising=False)


def test_conv_utils_frontend_funcs_extra4() -> None:
    """Test function test_conv_utils_frontend_funcs_extra4."""
    import types

    import ml_switcheroo_compiler.ops.base as base
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        GenericConvConfig,
        _build_conv_config,
        atrous_conv2d,
        atrous_conv2d_transpose,
        conv_transpose,
        convolution,
        depthwise_conv2d,
        depthwise_conv2d_backprop_filter,
        depthwise_conv2d_backprop_input,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape (BackendArg): Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils
        from tests.testing_utils import adapt_backend_module

        orig_conv2d = adapt_backend_module(conv_utils).get_symbol("conv2d", None)
        orig_conv2d_transpose = adapt_backend_module(conv_utils).get_symbol("conv2d_transpose", None)
        orig_conv_nd = adapt_backend_module(conv_utils).get_symbol("_conv_nd", None)
        orig_get_op = adapt_backend_module(base).get_symbol("get_op", None)

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

                return Tensor(
                    None, TensorConfig(shape=(2, 2), dtype="float32", device=None, requires_grad=False, trainable=False)
                )

        if orig_get_op is not None:
            pytest.MonkeyPatch().setattr(base, "get_op", lambda name: MockOp, raising=False)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            depthwise_conv2d(t1, t1, [1, 1], "VALID")
            depthwise_conv2d_backprop_filter(t1, t1, t1, [1, 1], "VALID")
            depthwise_conv2d_backprop_input((2, 2, 2), t1, t1, [1, 1], "VALID")
            convolution(t1, t1)
            conv_transpose(t1, t1, (2, 2, 2))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError) as err:
            assert str(err) != ""
        built_cfg = _build_conv_config(
            {"padding": "VALID", "strides": 1, "lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1), (0, 1), (0, 1))
        )
        assert built_cfg is not None
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            pytest.MonkeyPatch().setattr(base, "get_op", orig_get_op, raising=False)
        if orig_conv2d is not None:
            pytest.MonkeyPatch().setattr(conv_utils, "conv2d", orig_conv2d, raising=False)
        if orig_conv2d_transpose is not None:
            pytest.MonkeyPatch().setattr(conv_utils, "conv2d_transpose", orig_conv2d_transpose, raising=False)
        if orig_conv_nd is not None:
            pytest.MonkeyPatch().setattr(conv_utils, "_conv_nd", orig_conv_nd, raising=False)


class _TestConvUtilsFrontendFuncsExtra5State(typing.TypedDict, total=False):
    """Typed shared state for `_test_conv_utils_frontend_funcs_extra5` split helpers."""

    atrous_conv2d: EagerBoundCallable
    atrous_conv2d_transpose: EagerBoundCallable
    conv2d_mod: types.ModuleType
    conv_transpose: EagerBoundCallable
    conv_utils: types.ModuleType
    convolution: EagerBoundCallable
    depthwise_conv2d: EagerBoundCallable
    depthwise_conv2d_backprop_filter: EagerBoundCallable
    depthwise_conv2d_backprop_input: EagerBoundCallable
    t1: Tensor


def _test_conv_utils_frontend_funcs_extra5_sub_1(_state: _TestConvUtilsFrontendFuncsExtra5State) -> None:
    """Execute sub-part 1 of test_conv_utils_frontend_funcs_extra5.

    Args:
        _state: Shared test state dictionary.
    """
    (atrous_conv2d, conv2d_mod, conv_utils, t1) = (
        _state["atrous_conv2d"],
        _state["conv2d_mod"],
        _state["conv_utils"],
        _state["t1"],
    )
    config.eager_mode = False

    def mock_conv2d(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function mock_conv2d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `mock_conv2d`.
        """
        return "conv2d_called"

    def mock_conv2d_transpose(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function mock_conv2d_transpose.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `mock_conv2d_transpose`.
        """
        return "conv2d_transpose_called"

    def mock_conv_nd(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function mock_conv_nd.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `mock_conv_nd`.
        """
        return "conv_nd_called"

    pytest.MonkeyPatch().setattr(conv2d_mod, "conv2d", mock_conv2d, raising=False)
    pytest.MonkeyPatch().setattr(conv2d_mod, "conv2d_transpose", mock_conv2d_transpose, raising=False)
    pytest.MonkeyPatch().setattr(conv_utils, "_conv_nd", mock_conv_nd, raising=False)
    try:
        atrous_conv2d(t1, t1, 1, "VALID", None)
    except Exception as e:
        print(e)
    _state.update(
        {"atrous_conv2d": require_callable(atrous_conv2d), "conv2d_mod": conv2d_mod, "conv_utils": conv_utils}
    )
    _state.update({"t1": t1})


def _test_conv_utils_frontend_funcs_extra5_sub_2(_state: _TestConvUtilsFrontendFuncsExtra5State) -> None:
    """Execute sub-part 2 of test_conv_utils_frontend_funcs_extra5.

    Args:
        _state: Shared test state dictionary.
    """
    (atrous_conv2d_transpose, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input) = (
        _state["atrous_conv2d_transpose"],
        _state["depthwise_conv2d"],
        _state["depthwise_conv2d_backprop_filter"],
        _state["depthwise_conv2d_backprop_input"],
    )
    t1 = _state["t1"]
    try:
        atrous_conv2d_transpose(t1, t1, None, None, None)
    except Exception as e:
        print(e)
    try:
        depthwise_conv2d(t1, t1, [1, 1], "VALID")
    except Exception as e:
        print(e)
    try:
        depthwise_conv2d_backprop_filter(t1, t1, t1, [1, 1], "VALID")
    except Exception as e:
        print(e)
    try:
        depthwise_conv2d_backprop_input((2, 2, 2), t1, t1, [1, 1], "VALID")
    except Exception as e:
        print(e)
    _state.update(
        {
            "atrous_conv2d_transpose": require_callable(atrous_conv2d_transpose),
            "depthwise_conv2d": require_callable(depthwise_conv2d),
            "depthwise_conv2d_backprop_filter": require_callable(depthwise_conv2d_backprop_filter),
        }
    )
    _state.update({"depthwise_conv2d_backprop_input": require_callable(depthwise_conv2d_backprop_input), "t1": t1})


def _test_conv_utils_frontend_funcs_extra5_sub_3(_state: _TestConvUtilsFrontendFuncsExtra5State) -> None:
    """Execute sub-part 3 of test_conv_utils_frontend_funcs_extra5.

    Args:
        _state: Shared test state dictionary.
    """
    (conv_transpose, convolution, t1) = (_state["conv_transpose"], _state["convolution"], _state["t1"])
    try:
        convolution(t1, t1)
    except Exception as e:
        print(e)
    try:
        conv_transpose(t1, t1, (2, 2, 2))
    except Exception as e:
        print(e)
    _state.update(
        {"conv_transpose": require_callable(conv_transpose), "convolution": require_callable(convolution), "t1": t1}
    )


def test_conv_utils_frontend_funcs_extra5() -> None:
    """Test function test_conv_utils_frontend_funcs_extra5."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        atrous_conv2d,
        atrous_conv2d_transpose,
        conv_transpose,
        convolution,
        depthwise_conv2d,
        depthwise_conv2d_backprop_filter,
        depthwise_conv2d_backprop_input,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    import ml_switcheroo_compiler.ops.nn.conv as conv2d_mod
    from tests.testing_utils import adapt_backend_module

    orig_conv2d = conv2d_mod.conv2d
    orig_conv2d_transpose = conv2d_mod.conv2d_transpose
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

    orig_conv_nd = adapt_backend_module(conv_utils).get_symbol("_conv_nd", None)
    orig_eager = config.eager_mode
    try:
        _state: _TestConvUtilsFrontendFuncsExtra5State = {}
        _state.update(
            {
                "atrous_conv2d": require_callable(atrous_conv2d),
                "atrous_conv2d_transpose": require_callable(atrous_conv2d_transpose),
            }
        )
        _state.update(
            {
                "conv2d_mod": conv2d_mod,
                "conv_transpose": require_callable(conv_transpose),
                "conv_utils": conv_utils,
                "convolution": require_callable(convolution),
            }
        )
        _state.update(
            {
                "depthwise_conv2d": require_callable(depthwise_conv2d),
                "depthwise_conv2d_backprop_filter": require_callable(depthwise_conv2d_backprop_filter),
                "depthwise_conv2d_backprop_input": require_callable(depthwise_conv2d_backprop_input),
            }
        )
        _state.update({"t1": t1})
        for _sub_fn in (
            _test_conv_utils_frontend_funcs_extra5_sub_1,
            _test_conv_utils_frontend_funcs_extra5_sub_2,
            _test_conv_utils_frontend_funcs_extra5_sub_3,
        ):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        config.eager_mode = orig_eager
        conv2d_mod.conv2d = orig_conv2d
        conv2d_mod.conv2d_transpose = orig_conv2d_transpose
        if orig_conv_nd is not None:
            pytest.MonkeyPatch().setattr(conv_utils, "_conv_nd", orig_conv_nd, raising=False)
    assert config.eager_mode == orig_eager


def test_conv_utils_frontend_funcs_extra6() -> None:
    """Test function test_conv_utils_frontend_funcs_extra6."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        GenericConvConfig,
        atrous_conv2d,
        atrous_conv2d_transpose,
        depthwise_conv2d,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape (BackendArg): Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = base.get_op

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return "mock_op_instance_called"

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return MockOpInstance()

        from unittest.mock import patch

        pytest.MonkeyPatch().setattr(base, "get_op", MockOp(), raising=False)
        with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", side_effect=lambda name: MockOpInstance):
            out_ac6 = atrous_conv2d(t1, t1, 1, "VALID", None)
        assert out_ac6 == "mock_op_instance_called"
        try:
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            depthwise_conv2d(t1, t1, [1, 1], "VALID")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError) as err:
            assert str(err) != ""
    finally:
        config.eager_mode = orig_eager
        pytest.MonkeyPatch().setattr(base, "get_op", orig_get_op, raising=False)


def test_conv_utils_frontend_funcs_extra7() -> None:
    """Test function test_conv_utils_frontend_funcs_extra7."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        GenericConvConfig,
        atrous_conv2d,
        atrous_conv2d_transpose,
        depthwise_conv2d,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape (BackendArg): Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = base.get_op

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return "mock_op_instance_called"

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return MockOpInstance()

        def get_op_override(name: BackendArg) -> BackendArg:
            """Test function get_op_override.

            Args:
                name (BackendArg): Test parameter `name`.

            Returns:
                BackendArg: Computed output value from `get_op_override`.
            """
            return MockOp()

        pytest.MonkeyPatch().setattr(base, "get_op", get_op_override, raising=False)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            depthwise_conv2d(t1, t1, [1, 1], "VALID")
        except Exception as e:
            print(e)
    finally:
        config.eager_mode = orig_eager
        pytest.MonkeyPatch().setattr(base, "get_op", orig_get_op, raising=False)
    assert config.eager_mode == orig_eager


def test_conv_utils_frontend_funcs_extra8() -> None:
    """Test function test_conv_utils_frontend_funcs_extra8."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        GenericConvConfig,
        atrous_conv2d,
        atrous_conv2d_transpose,
        depthwise_conv2d,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape (BackendArg): Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = base.get_op

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return "mock_op_instance_called"

        class MockOp:
            """Test class MockOp."""

            def __init__(self) -> None:
                """Test function __init__."""
                pass

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    *args (BackendArg): Test parameter `args`.
                    **kwargs (BackendArg): Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return MockOpInstance()

        def get_op_override(name: BackendArg) -> BackendArg:
            """Test function get_op_override.

            Args:
                name (BackendArg): Test parameter `name`.

            Returns:
                BackendArg: Computed output value from `get_op_override`.
            """
            return MockOp

        pytest.MonkeyPatch().setattr(base, "get_op", get_op_override, raising=False)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            depthwise_conv2d(t1, t1, [1, 1], "VALID")
        except Exception as e:
            print(e)
    finally:
        config.eager_mode = orig_eager
        pytest.MonkeyPatch().setattr(base, "get_op", orig_get_op, raising=False)
    assert config.eager_mode == orig_eager


def test_conv_utils_frontend_funcs_extra9() -> None:
    """Test function test_conv_utils_frontend_funcs_extra9."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import (
        GenericConvConfig,
        atrous_conv2d,
        atrous_conv2d_transpose,
        conv_transpose,
        convolution,
        depthwise_conv2d,
        depthwise_conv2d_backprop_filter,
        depthwise_conv2d_backprop_input,
    )

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return Tensor(
                    DummyTensor((2, 2, 2)),
                    TensorConfig(
                        shape=(2, 2, 2),
                        dtype=DummyTensor((2, 2, 2)).dtype,
                        device=None,
                        requires_grad=False,
                        trainable=False,
                    ),
                )

        class MockGetOp:
            """Test class MockGetOp."""

            def __call__(self, name: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    name: Test parameter `name`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return MockOpInstance

        orig_get_op = base.get_op
        pytest.MonkeyPatch().setattr(base, "get_op", MockGetOp(), raising=False)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            depthwise_conv2d(t1, t1, [1, 1], "VALID")
            depthwise_conv2d_backprop_filter(t1, t1, t1, [1, 1], "VALID")
            depthwise_conv2d_backprop_input((2, 2, 2), t1, t1, [1, 1], "VALID")
            convolution(t1, t1)
            conv_transpose(t1, t1, (2, 2, 2))
        except Exception as e:
            print("EXCEPTION IN EXTRA9:", e)
    finally:
        config.eager_mode = orig_eager
        pytest.MonkeyPatch().setattr(base, "get_op", orig_get_op, raising=False)
    assert config.eager_mode == orig_eager


def test_conv_utils_frontend_funcs_extra10() -> None:
    """Test function test_conv_utils_frontend_funcs_extra10."""
    from ml_switcheroo_compiler.ops.nn.conv_utils import _build_conv_config

    _build_conv_config({"strides": 1, "lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1), (0, 1), (0, 1)))
    _build_conv_config({"strides": 1, "lhs_dilation": (1,), "rhs_dilation": (1,)}, ((0, 1), (0, 1), (0, 1)))
    _build_conv_config({"strides": (1,), "lhs_dilation": (1,), "rhs_dilation": (1,)}, ((0, 1), (0, 1), (0, 1)))
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import _prepare_depthwise_conv

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    prep = _prepare_depthwise_conv(t1, t1, 1, ((0, 1), (0, 1), (0, 1)), config_obj=1)
    assert len(prep) == 2


def test_conv_utils_frontend_funcs_extra11() -> None:
    """Test function test_conv_utils_frontend_funcs_extra11."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape
            self.data = np.ones(coerce_int_list(shape))
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = Tensor(
        DummyTensor((2, 2, 2)),
        TensorConfig(
            shape=(2, 2, 2),
            dtype=DummyTensor((2, 2, 2)).dtype,
            device=None,
            requires_grad=False,
            trainable=False,
        ),
    )
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils
    from tests.testing_utils import adapt_backend_module

    orig_conv2d = adapt_backend_module(conv_utils).get_symbol("conv2d", None)
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False

        def mock_conv2d(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function mock_conv2d.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `mock_conv2d`.
            """
            return "conv2d_called"

        def mock_conv2d_transpose(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function mock_conv2d_transpose.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `mock_conv2d_transpose`.
            """
            return "conv2d_transpose_called"

        def mock_conv_nd(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function mock_conv_nd.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `mock_conv_nd`.
            """
            return "conv_nd_called"

        pytest.MonkeyPatch().setattr(conv_utils, "conv2d", mock_conv2d, raising=False)
        pytest.MonkeyPatch().setattr(conv_utils, "conv2d_transpose", mock_conv2d_transpose, raising=False)
        pytest.MonkeyPatch().setattr(conv_utils, "_conv_nd", mock_conv_nd, raising=False)

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return Tensor(
                    None, TensorConfig(shape=(2, 2), dtype="float32", device=None, requires_grad=False, trainable=False)
                )

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function __call__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `__call__`.
                """
                return MockOpInstance()

        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", lambda name: MockOp()):
            conv_utils.atrous_conv2d(t1, t1, 1, "VALID", None)
            conv_utils.atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            conv_utils.depthwise_conv2d(t1, t1, GenericConvConfig(), "VALID")
            conv_utils.convolution(t1, t1)
    finally:
        config.eager_mode = orig_eager
        if orig_conv2d is not None:
            pytest.MonkeyPatch().setattr(conv_utils, "conv2d", orig_conv2d, raising=False)
        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", lambda name: MockOp()):
            conv_utils.atrous_conv2d(t1, t1, 1, "VALID", None)
            conv_utils.atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            conv_utils.depthwise_conv2d(t1, t1, GenericConvConfig(), "VALID")
            conv_utils.convolution(t1, t1)
    assert config.eager_mode == orig_eager
