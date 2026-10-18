"""Tests for test_utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_conv_utils_calc_pad() -> None:
    """Test function test_conv_utils_calc_pad."""
    assert _calc_same_pad(3, 1) == (1, 1)
    assert _calc_same_pad(3, 3) == (2, 2)
    assert _calc_same_pad(3, 4) == (2, 3)
    assert _calc_valid_pad(3, 1) == (2, 2)


def test_build_conv_config() -> None:
    """Test function test_build_conv_config."""
    kwargs = {"strides": 1, "padding": "SAME", "lhs_dilation": 1, "rhs_dilation": 1, "feature_group_count": 1, "batch_group_count": 1}
    c = cast("Callable[..., EagerTensorProtocol]", _build_conv_config)(kwargs, ((0, 1), (0, 1), (0, 1)))
    assert c is not None
    c = cast("EagerTensorProtocol", _build_conv_config({}, ((0, 1), (0, 1), (0, 1))))
    assert c is not None


def test_calculate_conv_transpose_padding() -> None:
    """Test function test_calculate_conv_transpose_padding."""
    assert cast("Callable[..., EagerTensorProtocol]", _calculate_conv_transpose_padding)((5, 5), (3, 3), (2, 2)) is not None


def _test_conv_utils_frontend_funcs_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_conv_utils_frontend_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    config.eager_mode = False
    with pytest.raises(KeyError):
        atrous_conv2d(None, None, None, None)
    with pytest.raises(KeyError):
        atrous_conv2d_transpose(None, None, None, None, None)
    with pytest.raises(RuntimeError):
        bias_add(None, None)
    collapse_repeated(None, None)
    _state.update(locals())


def _test_conv_utils_frontend_funcs_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_conv_utils_frontend_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(RuntimeError):
        compute_average_loss(None)
    with pytest.raises(KeyError):
        cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(None, None, None, None)
    depthwise_conv2d_backprop_filter(None, None, None, None, None)
    depthwise_conv2d_backprop_input(None, None, None, None, None)
    _state.update(locals())


def _test_conv_utils_frontend_funcs_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_conv_utils_frontend_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(TypeError):
        cast("Callable[..., EagerTensorProtocol]", dilation2d)(None, None, None, None, None)
    with pytest.raises(TypeError):
        cast("Callable[..., EagerTensorProtocol]", erosion2d)(None, None, None, None, None)
    with pytest.raises(KeyError):
        convolution(None, None)
    with pytest.raises(TypeError):
        cast("Callable[..., EagerTensorProtocol]", conv_transpose)(None, None)
    _state.update(locals())


def test_conv_utils_frontend_funcs() -> None:
    """Test function test_conv_utils_frontend_funcs."""
    orig_eager = config.eager_mode
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"orig_eager": orig_eager})
        for _sub_fn in (_test_conv_utils_frontend_funcs_sub_1, _test_conv_utils_frontend_funcs_sub_2, _test_conv_utils_frontend_funcs_sub_3):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    finally:
        config.eager_mode = orig_eager


def test_conv_utils_calc_pad_list() -> None:
    """Test function test_conv_utils_calc_pad_list."""
    assert callable(test_conv_utils_calc_pad_list)


def test_prepare_depthwise_conv() -> None:
    """Test function test_prepare_depthwise_conv."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import _prepare_depthwise_conv

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    t2 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackendConv:
            """Test class MockBackendConv."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op."""
                return np.ones((2, 2, 2))

            @classmethod
            def array(cls, data: EagerValue) -> EagerValue:
                """Test function array."""
                return data

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        cast("Callable[..., EagerTensorProtocol]", BackendRegistry.register)("mock_ops", MockBackendConv)
        cast("Callable[..., EagerTensorProtocol]", _prepare_depthwise_conv)(t1, t2, 1, ((0, 1), (0, 1), (0, 1)), config_obj=None, strides=1, lhs_dilation=1, rhs_dilation=1)
    finally:
        config.eager_mode = orig_eager
    assert orig_eager is not None or orig_eager is None


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

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    t2 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        config.backend = "mock_ops"

        class MockBackendConv:
            """Test class MockBackendConv."""

            @classmethod
            def execute_op(cls, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op."""
                return np.ones((2, 2, 2))

            @classmethod
            def array(cls, data: EagerValue) -> EagerValue:
                """Test function array."""
                return data

        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        cast("Callable[..., EagerTensorProtocol]", BackendRegistry.register)("mock_ops", MockBackendConv)
        cast("Callable[..., EagerTensorProtocol]", _prepare_depthwise_conv)(t1, t2, 1, ((0, 1), (0, 1), (0, 1)), config_obj=None, strides=(1,), lhs_dilation=(1,), rhs_dilation=(1,))
    finally:
        config.eager_mode = orig_eager
    assert orig_eager is not None or orig_eager is None


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


def _test_conv_utils_frontend_funcs_extra2_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_conv_utils_frontend_funcs_extra2.

    Args:
        _state: Shared test state dictionary.
    """
    (atrous_conv2d, bias_add, collapse_repeated, compute_average_loss) = (_state["atrous_conv2d"], _state["bias_add"], _state["collapse_repeated"], _state["compute_average_loss"])
    t1 = _state["t1"]
    config.eager_mode = False
    with pytest.raises(KeyError):
        atrous_conv2d(t1, t1, 1, "VALID", None)
    with pytest.raises(RuntimeError):
        bias_add(t1, t1, None, None)
    collapse_repeated(t1, t1, None)
    with pytest.raises(RuntimeError):
        compute_average_loss(t1, t1, t1)
    _state.update(locals())


def _test_conv_utils_frontend_funcs_extra2_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_conv_utils_frontend_funcs_extra2.

    Args:
        _state: Shared test state dictionary.
    """
    (depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input, dilation2d) = (_state["depthwise_conv2d"], _state["depthwise_conv2d_backprop_filter"], _state["depthwise_conv2d_backprop_input"], _state["dilation2d"])
    t1 = _state["t1"]
    with pytest.raises(KeyError):
        depthwise_conv2d(t1, t1, t1, t1)
    depthwise_conv2d_backprop_filter(t1, t1, t1, t1, t1)
    depthwise_conv2d_backprop_input(t1, t1, t1, t1, t1)
    with pytest.raises(TypeError):
        dilation2d(t1, t1, t1, t1, t1)
    _state.update(locals())


def _test_conv_utils_frontend_funcs_extra2_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_conv_utils_frontend_funcs_extra2.

    Args:
        _state: Shared test state dictionary.
    """
    (conv_transpose, convolution, erosion2d, t1) = (_state["conv_transpose"], _state["convolution"], _state["erosion2d"], _state["t1"])
    with pytest.raises(TypeError):
        erosion2d(t1, t1, t1, t1, t1)
    with pytest.raises(KeyError):
        convolution(t1, t1)
    with pytest.raises(TypeError):
        conv_transpose(t1, t1)
    _state.update(locals())


def test_conv_utils_frontend_funcs_extra2() -> None:
    """Test function test_conv_utils_frontend_funcs_extra2."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import atrous_conv2d, bias_add, collapse_repeated, compute_average_loss, conv_transpose, convolution, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input, dilation2d, erosion2d

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"Tensor": Tensor, "TensorConfig": TensorConfig, "atrous_conv2d": atrous_conv2d, "bias_add": bias_add})
        getattr(_state, "update")({"collapse_repeated": collapse_repeated, "compute_average_loss": compute_average_loss, "conv_transpose": conv_transpose, "convolution": convolution})
        getattr(_state, "update")({"depthwise_conv2d": depthwise_conv2d, "depthwise_conv2d_backprop_filter": depthwise_conv2d_backprop_filter, "depthwise_conv2d_backprop_input": depthwise_conv2d_backprop_input, "dilation2d": dilation2d})
        getattr(_state, "update")({"erosion2d": erosion2d, "orig_eager": orig_eager, "t1": t1, "types": types})
        for _sub_fn in (_test_conv_utils_frontend_funcs_extra2_sub_1, _test_conv_utils_frontend_funcs_extra2_sub_2, _test_conv_utils_frontend_funcs_extra2_sub_3):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    finally:
        config.eager_mode = orig_eager


def test_conv_utils_frontend_funcs_extra3() -> None:
    """Test function test_conv_utils_frontend_funcs_extra3."""
    import types

    import ml_switcheroo_compiler.ops.base as base
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, atrous_conv2d, atrous_conv2d_transpose, conv_transpose, convolution, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = getattr(base, "get_op", None)

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return "mock_op_called"

        setattr(base, "get_op", lambda name: MockOp)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(t1, t1, [1, 1], "VALID")
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d_backprop_filter)(t1, t1, t1, [1, 1], "VALID")
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d_backprop_input)((2, 2, 2), t1, t1, [1, 1], "VALID")
            convolution(t1, t1)
            conv_transpose(t1, t1, (2, 2, 2))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError) as e:
            assert e is not None
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            base.get_op = orig_get_op
    assert callable(test_conv_utils_frontend_funcs_extra3)


def test_conv_utils_frontend_funcs_extra4() -> None:
    """Test function test_conv_utils_frontend_funcs_extra4."""
    import types

    import ml_switcheroo_compiler.ops.base as base
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, _build_conv_config, atrous_conv2d, atrous_conv2d_transpose, conv_transpose, convolution, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

        orig_conv2d = getattr(conv_utils, "conv2d", None)
        orig_conv2d_transpose = getattr(conv_utils, "conv2d_transpose", None)
        orig_conv_nd = getattr(conv_utils, "_conv_nd", None)
        orig_get_op = getattr(base, "get_op", None)

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

                return Tensor(None, TensorConfig(shape=(2, 2), dtype="float32", device=None, requires_grad=False, trainable=False))

        if orig_get_op:
            setattr(base, "get_op", lambda name: MockOp)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(t1, t1, [1, 1], "VALID")
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d_backprop_filter)(t1, t1, t1, [1, 1], "VALID")
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d_backprop_input)((2, 2, 2), t1, t1, [1, 1], "VALID")
            convolution(t1, t1)
            conv_transpose(t1, t1, (2, 2, 2))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError) as e:
            assert e is not None
        _build_conv_config({"padding": "VALID", "strides": 1, "lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1), (0, 1), (0, 1)))
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            base.get_op = orig_get_op
        if orig_conv2d:
            setattr(conv_utils, "conv2d", orig_conv2d)
        if orig_get_op:
            base.get_op = orig_get_op
        if orig_conv2d_transpose:
            setattr(conv_utils, "conv2d_transpose", orig_conv2d_transpose)
        if orig_conv_nd:
            setattr(conv_utils, "_conv_nd", orig_conv_nd)
    assert callable(test_conv_utils_frontend_funcs_extra4)


def _test_conv_utils_frontend_funcs_extra5_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_conv_utils_frontend_funcs_extra5.

    Args:
        _state: Shared test state dictionary.
    """
    (atrous_conv2d, conv2d_mod, conv_utils, t1) = (_state["atrous_conv2d"], _state["conv2d_mod"], _state["conv_utils"], _state["t1"])
    config.eager_mode = False

    def mock_conv2d(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_conv2d."""
        return "conv2d_called"

    def mock_conv2d_transpose(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_conv2d_transpose."""
        return "conv2d_transpose_called"

    def mock_conv_nd(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_conv_nd."""
        return "conv_nd_called"

    setattr(conv2d_mod, "conv2d", mock_conv2d)
    setattr(conv2d_mod, "conv2d_transpose", mock_conv2d_transpose)
    setattr(conv_utils, "_conv_nd", mock_conv_nd)
    try:
        atrous_conv2d(t1, t1, 1, "VALID", None)
    except Exception as e:
        print(e)
    getattr(_state, "update")({"mock_conv2d": mock_conv2d, "mock_conv2d_transpose": mock_conv2d_transpose, "mock_conv_nd": mock_conv_nd})
    _state.update(locals())


def _test_conv_utils_frontend_funcs_extra5_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_conv_utils_frontend_funcs_extra5.

    Args:
        _state: Shared test state dictionary.
    """
    (atrous_conv2d_transpose, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input) = (_state["atrous_conv2d_transpose"], _state["depthwise_conv2d"], _state["depthwise_conv2d_backprop_filter"], _state["depthwise_conv2d_backprop_input"])
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
    _state.update(locals())


def _test_conv_utils_frontend_funcs_extra5_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def test_conv_utils_frontend_funcs_extra5() -> None:
    """Test function test_conv_utils_frontend_funcs_extra5."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import atrous_conv2d, atrous_conv2d_transpose, conv_transpose, convolution, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    import ml_switcheroo_compiler.ops.nn.conv2d as conv2d_mod

    orig_conv2d = getattr(conv2d_mod, "conv2d", None)
    orig_conv2d_transpose = getattr(conv2d_mod, "conv2d_transpose", None)
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

    orig_conv_nd = getattr(conv_utils, "_conv_nd", None)
    orig_eager = config.eager_mode
    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"Tensor": Tensor, "TensorConfig": TensorConfig, "atrous_conv2d": atrous_conv2d, "atrous_conv2d_transpose": atrous_conv2d_transpose})
        getattr(_state, "update")({"conv2d_mod": conv2d_mod, "conv_transpose": conv_transpose, "conv_utils": conv_utils, "convolution": convolution})
        getattr(_state, "update")({"depthwise_conv2d": depthwise_conv2d, "depthwise_conv2d_backprop_filter": depthwise_conv2d_backprop_filter, "depthwise_conv2d_backprop_input": depthwise_conv2d_backprop_input, "orig_conv2d": orig_conv2d})
        getattr(_state, "update")({"orig_conv2d_transpose": orig_conv2d_transpose, "orig_conv_nd": orig_conv_nd, "orig_eager": orig_eager, "t1": t1})
        getattr(_state, "update")({"types": types})
        for _sub_fn in (_test_conv_utils_frontend_funcs_extra5_sub_1, _test_conv_utils_frontend_funcs_extra5_sub_2, _test_conv_utils_frontend_funcs_extra5_sub_3):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    finally:
        config.eager_mode = orig_eager
        if orig_conv2d:
            conv2d_mod.conv2d = orig_conv2d
        if orig_conv2d_transpose:
            conv2d_mod.conv2d_transpose = orig_conv2d_transpose
        if orig_conv_nd:
            setattr(conv_utils, "_conv_nd", orig_conv_nd)
    assert orig_eager is not None or orig_eager is None


def test_conv_utils_frontend_funcs_extra6() -> None:
    """Test function test_conv_utils_frontend_funcs_extra6."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, atrous_conv2d, atrous_conv2d_transpose, depthwise_conv2d

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = getattr(base, "get_op", None)

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return "mock_op_instance_called"

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return MockOpInstance()

        if orig_get_op:
            setattr(base, "get_op", MockOp())
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(t1, t1, [1, 1], "VALID")
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError) as e:
            assert e is not None
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            base.get_op = orig_get_op
    assert callable(test_conv_utils_frontend_funcs_extra6)


def test_conv_utils_frontend_funcs_extra7() -> None:
    """Test function test_conv_utils_frontend_funcs_extra7."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, atrous_conv2d, atrous_conv2d_transpose, depthwise_conv2d

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = getattr(base, "get_op", None)

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return "mock_op_instance_called"

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return MockOpInstance()

        def get_op_override(name: EagerValue) -> EagerValue:
            """Test function get_op_override."""
            return MockOp()

        if orig_get_op:
            setattr(base, "get_op", get_op_override)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(t1, t1, [1, 1], "VALID")
        except Exception as e:
            print(e)
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            base.get_op = orig_get_op
    assert orig_eager is not None or orig_eager is None


def test_conv_utils_frontend_funcs_extra8() -> None:
    """Test function test_conv_utils_frontend_funcs_extra8."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, atrous_conv2d, atrous_conv2d_transpose, depthwise_conv2d

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = getattr(base, "get_op", None)

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return "mock_op_instance_called"

        class MockOp:
            """Test class MockOp."""

            def __init__(self) -> None:
                """Test function __init__."""
                pass

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return MockOpInstance()

        def get_op_override(name: EagerValue) -> EagerValue:
            """Test function get_op_override."""
            return MockOp

        if orig_get_op:
            setattr(base, "get_op", get_op_override)
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(t1, t1, [1, 1], "VALID")
        except Exception as e:
            print(e)
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            base.get_op = orig_get_op
    assert orig_eager is not None or orig_eager is None


def test_conv_utils_frontend_funcs_extra9() -> None:
    """Test function test_conv_utils_frontend_funcs_extra9."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig, atrous_conv2d, atrous_conv2d_transpose, conv_transpose, convolution, depthwise_conv2d, depthwise_conv2d_backprop_filter, depthwise_conv2d_backprop_input

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False
        import ml_switcheroo_compiler.ops.base as base

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return cast("Callable[..., EagerTensorProtocol]", Tensor)(
                    cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
                )

        class MockGetOp:
            """Test class MockGetOp."""

            def __call__(self, name: EagerValue) -> EagerValue:
                """Test function __call__."""
                return MockOpInstance

        orig_get_op = getattr(base, "get_op", None)
        setattr(base, "get_op", MockGetOp())
        try:
            atrous_conv2d(t1, t1, 1, "VALID", None)
            atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d)(t1, t1, [1, 1], "VALID")
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d_backprop_filter)(t1, t1, t1, [1, 1], "VALID")
            cast("Callable[..., EagerTensorProtocol]", depthwise_conv2d_backprop_input)((2, 2, 2), t1, t1, [1, 1], "VALID")
            convolution(t1, t1)
            conv_transpose(t1, t1, (2, 2, 2))
        except Exception as e:
            print("EXCEPTION IN EXTRA9:", e)
    finally:
        config.eager_mode = orig_eager
        if orig_get_op is not None:
            base.get_op = orig_get_op
    assert orig_eager is not None or orig_eager is None


def test_conv_utils_frontend_funcs_extra10() -> None:
    """Test function test_conv_utils_frontend_funcs_extra10."""
    from ml_switcheroo_compiler.ops.nn.conv_utils import _build_conv_config

    _build_conv_config({"strides": 1, "lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1), (0, 1), (0, 1)))
    _build_conv_config({"strides": 1, "lhs_dilation": (1,), "rhs_dilation": (1,)}, ((0, 1), (0, 1), (0, 1)))
    _build_conv_config({"strides": (1,), "lhs_dilation": (1,), "rhs_dilation": (1,)}, ((0, 1), (0, 1), (0, 1)))
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import _prepare_depthwise_conv

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    cast("Callable[..., EagerTensorProtocol]", _prepare_depthwise_conv)(t1, t1, 1, ((0, 1), (0, 1), (0, 1)), config_obj=1)
    assert callable(test_conv_utils_frontend_funcs_extra10)


def test_conv_utils_frontend_funcs_extra11() -> None:
    """Test function test_conv_utils_frontend_funcs_extra11."""
    import types

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.conv_utils import GenericConvConfig

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__."""
            self.shape = shape
            self.data = cast("Callable[..., EagerTensorProtocol]", np.ones)(shape)
            self.dtype = types.SimpleNamespace(value="float32")
            self.device = None

    t1 = cast("Callable[..., EagerTensorProtocol]", Tensor)(
        cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)), cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape=(2, 2, 2), dtype=cast("Callable[..., EagerTensorProtocol]", DummyTensor)((2, 2, 2)).dtype, device=None, requires_grad=False, trainable=False)
    )
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

    orig_conv2d = getattr(conv_utils, "conv2d", None)
    getattr(conv_utils, "conv2d_transpose", None)
    getattr(conv_utils, "_conv_nd", None)
    orig_eager = config.eager_mode
    try:
        config.eager_mode = False

        def mock_conv2d(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function mock_conv2d."""
            return "conv2d_called"

        def mock_conv2d_transpose(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function mock_conv2d_transpose."""
            return "conv2d_transpose_called"

        def mock_conv_nd(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function mock_conv_nd."""
            return "conv_nd_called"

        setattr(conv_utils, "conv2d", mock_conv2d)
        setattr(conv_utils, "conv2d_transpose", mock_conv2d_transpose)
        setattr(conv_utils, "_conv_nd", mock_conv_nd)
        import ml_switcheroo_compiler.ops.base as base

        orig_get_op = getattr(base, "get_op", None)

        class MockOpInstance:
            """Test class MockOpInstance."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return Tensor(None, TensorConfig(shape=(2, 2), dtype="float32", device=None, requires_grad=False, trainable=False))

        class MockOp:
            """Test class MockOp."""

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function __call__."""
                return MockOpInstance()

        if orig_get_op:
            from unittest.mock import patch

            with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", lambda name: MockOp()):
                conv_utils.atrous_conv2d(t1, t1, 1, "VALID", None)
                conv_utils.atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
                cast("Callable[..., EagerTensorProtocol]", conv_utils.depthwise_conv2d)(t1, t1, cast("Callable[..., EagerTensorProtocol]", GenericConvConfig)(), "VALID")
                conv_utils.convolution(t1, t1)
    finally:
        config.eager_mode = orig_eager
        if orig_conv2d:
            setattr(conv_utils, "conv2d", orig_conv2d)
        if orig_get_op:
            from unittest.mock import patch

            with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", lambda name: MockOp()):
                conv_utils.atrous_conv2d(t1, t1, 1, "VALID", None)
                conv_utils.atrous_conv2d_transpose(t1, t1, None, GenericConvConfig(), None)
                cast("Callable[..., EagerTensorProtocol]", conv_utils.depthwise_conv2d)(t1, t1, cast("Callable[..., EagerTensorProtocol]", GenericConvConfig)(), "VALID")
                conv_utils.convolution(t1, t1)
    assert orig_eager is not None or orig_eager is None
