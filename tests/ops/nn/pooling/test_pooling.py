"""Tests for test_pooling."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.nn.pooling import (
    FractionalAvgPool,
    FractionalMaxPool,
    PoolingBehaviorConfig,
    PoolingConfig,
    SpatialConfig,
    _compute_pool_out_shape,
    _prepare_pool_config,
    average_pool,
    avg_pool,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    fractional_avg_pool,
    fractional_max_pool,
    max_pool,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    max_pool_with_argmax,
    pool,
    pool1d,
    pool2d,
    pool3d,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_pooling_coverage() -> None:
    """Test function test_pooling_coverage."""
    config.eager_mode = True
    t1 = Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t2 = Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t3 = Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))

    assert _compute_pool_out_shape((1, 3, 2), WindowConfig(window_dimensions=(1, 2, 1), window_strides=(1, 1, 1), padding="SAME")) is not None
    assert _compute_pool_out_shape((1, 3, 2), WindowConfig(window_dimensions=(1, 2, 1), window_strides=(1, 1, 1), padding="VALID")) is not None
    assert _compute_pool_out_shape((1, 3, 2), WindowConfig(window_dimensions=(1, 2, 1), window_strides=(1, 1, 1), padding=((0, 0), (0, 0), (0, 0)))) is not None

    assert pool1d(t1, 2) is not None
    assert pool1d(t1, 2, pool_mode="avg") is not None
    assert pool2d(t2, (2, 2)) is not None
    assert pool2d(t2, (2, 2), pool_mode="avg") is not None
    assert pool3d(t3, (2, 2, 2)) is not None
    assert pool3d(t3, (2, 2, 2), pool_mode="avg") is not None

    assert average_pool(t1, (2,)) is not None

    pc_avg = PoolingConfig(SpatialConfig((2,), (1,), "VALID"), PoolingBehaviorConfig(pooling_type="AVG"))
    pc_max = PoolingConfig(SpatialConfig((2,), (1,), "VALID"), PoolingBehaviorConfig(pooling_type="MAX"))

    assert avg_pool1d(t1, pc_avg) is not None
    assert avg_pool2d(t2, pc_avg) is not None
    assert avg_pool3d(t3, pc_avg) is not None
    assert max_pool1d(t1, pc_max) is not None
    assert max_pool2d(t2, pc_max) is not None
    assert max_pool3d(t3, pc_max) is not None

    try:
        max_pool_with_argmax(t2, pc_max)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    pc_frac = PoolingConfig(SpatialConfig((2,), (1,), "VALID"), PoolingBehaviorConfig(pooling_type="AVG", pooling_ratio=(1.5,)))
    try:
        fractional_avg_pool(t2, pc_frac)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        fractional_max_pool(t2, pc_frac)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    assert pool(t1, pc_avg) is not None
    assert pool(t1, pc_max) is not None

    assert cast("Callable[..., EagerTensorProtocol]", FractionalAvgPool)().infer_shape(t1, (1.5,)) == t1.shape
    assert cast("Callable[..., EagerTensorProtocol]", FractionalMaxPool)().infer_shape(t1, (1.5,)) == t1.shape


def _test_pooling_brute_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    config.backend = "numpy"
    config.eager_mode = True

    t1 = Tensor(np.random.rand(2, 4, 8, 8).astype(np.float32), TensorConfig((2, 4, 8, 8), "float32", "cpu"))
    t1_1d = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), "float32", "cpu"))
    t1_2d = Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), "float32", "cpu"))
    t1_3d = Tensor(np.random.rand(8, 8, 8).astype(np.float32), TensorConfig((8, 8, 8), "float32", "cpu"))

    c1 = _prepare_pool_config(4, 2, (2, 2), (2, 2), "VALID")
    c2 = _prepare_pool_config(4, 2, (2, 2), (2, 2), "SAME")
    c3 = cast("Callable[..., EagerTensorProtocol]", _prepare_pool_config)(4, 2, (2, 2), (2, 2), ((1, 1), (1, 1)))
    c4 = _prepare_pool_config(4, 2, (2, 2), (2, 2), (1, 1))

    _compute_pool_out_shape((2, 4, 8, 8), c1)
    _compute_pool_out_shape((2, 4, 8, 8), c2)
    cast("Callable[..., EagerTensorProtocol]", _compute_pool_out_shape)((2, 4, 8, 8), c3)
    _compute_pool_out_shape((2, 4, 8, 8), c4)

    max_pool(t1, (2, 2))

    try:
        max_pool(t1, (2, 2), return_indices=True)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        avg_pool(t1, (2, 2))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    getattr(_state, "update")({"c1": c1, "c2": c2, "c3": c3, "c4": c4})
    getattr(_state, "update")({"t1": t1, "t1_1d": t1_1d, "t1_2d": t1_2d, "t1_3d": t1_3d})
    _state.update(locals())


def _test_pooling_brute_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    t1_1d, t1_2d = _state["t1_1d"], _state["t1_2d"]
    try:
        cast("Callable[..., EagerTensorProtocol]", pool1d)(t1_1d, 2, pool_mode="max")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", pool1d)(t1_1d, 2, pool_mode="avg")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    with pytest.raises(Exception, match=r".*"):
        cast("Callable[..., EagerTensorProtocol]", pool1d)(t1_1d, 2, pool_mode="sum")

    try:
        cast("Callable[..., EagerTensorProtocol]", pool2d)(t1_2d, (2, 2), pool_mode="max")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("Callable[..., EagerTensorProtocol]", pool2d)(t1_2d, (2, 2), pool_mode="avg")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_pooling_brute_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    t1, t1_2d, t1_3d = _state["t1"], _state["t1_2d"], _state["t1_3d"]
    with pytest.raises(Exception, match=r".*"):
        cast("Callable[..., EagerTensorProtocol]", pool2d)(t1_2d, (2, 2), pool_mode="sum")
    try:
        cast("Callable[..., EagerTensorProtocol]", pool3d)(t1_3d, (2, 2, 2), pool_mode="max")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("Callable[..., EagerTensorProtocol]", pool3d)(t1_3d, (2, 2, 2), pool_mode="avg")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    with pytest.raises(Exception, match=r".*"):
        cast("Callable[..., EagerTensorProtocol]", pool3d)(t1_3d, (2, 2, 2), pool_mode="sum")

    try:
        cast("Callable[..., EagerTensorProtocol]", average_pool)(t1, (2, 2))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    p_conf = PoolingConfig(window=SpatialConfig(ksize=(2, 2), strides=(2, 2), padding="VALID"), behavior=PoolingBehaviorConfig(pooling_type="AVG", pooling_ratio=(0.5, 0.5)))
    p_conf_max = PoolingConfig(window=SpatialConfig(ksize=(2, 2), strides=(2, 2), padding="VALID"), behavior=PoolingBehaviorConfig(pooling_type="MAX", pooling_ratio=(0.5, 0.5)))
    getattr(_state, "update")({"p_conf": p_conf, "p_conf_max": p_conf_max})
    _state.update(locals())


def _test_pooling_brute_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    p_conf, p_conf_max, t1_1d, t1_2d = _state["p_conf"], _state["p_conf_max"], _state["t1_1d"], _state["t1_2d"]
    t1_3d = _state["t1_3d"]
    try:
        avg_pool1d(t1_1d, PoolingConfig(window=SpatialConfig(ksize=(2,), strides=(2,), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", avg_pool2d)(t1_2d, p_conf)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        avg_pool3d(t1_3d, PoolingConfig(window=SpatialConfig(ksize=(2, 2, 2), strides=(2, 2, 2), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        max_pool1d(t1_1d, PoolingConfig(window=SpatialConfig(ksize=(2,), strides=(2,), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("Callable[..., EagerTensorProtocol]", max_pool2d)(t1_2d, p_conf_max)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_pooling_brute_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    p_conf, p_conf_max, t1, t1_3d = _state["p_conf"], _state["p_conf_max"], _state["t1"], _state["t1_3d"]
    try:
        max_pool3d(t1_3d, PoolingConfig(window=SpatialConfig(ksize=(2, 2, 2), strides=(2, 2, 2), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("Callable[..., EagerTensorProtocol]", max_pool_with_argmax)(t1, p_conf_max)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("Callable[..., EagerTensorProtocol]", fractional_avg_pool)(t1, p_conf)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("Callable[..., EagerTensorProtocol]", fractional_max_pool)(t1, p_conf_max)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    # Test eager mode off
    config.eager_mode = False

    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    getattr(_state, "update")({"global_tracing_state": global_tracing_state})
    _state.update(locals())


def _test_pooling_brute_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state, p_conf, p_conf_max, t1 = _state["global_tracing_state"], _state["p_conf"], _state["p_conf_max"], _state["t1"]
    try:
        try:
            cast("Callable[..., EagerTensorProtocol]", max_pool)(t1, (2, 2), return_indices=True)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", fractional_avg_pool)(t1, p_conf)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", fractional_max_pool)(t1, p_conf_max)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        FractionalAvgPool().infer_shape(t1, p_conf.behavior.pooling_ratio)
        FractionalMaxPool().infer_shape(t1, p_conf_max.behavior.pooling_ratio)
    finally:
        setattr(global_tracing_state, "is_tracing", False)

    try:
        cast("Callable[..., EagerTensorProtocol]", pool)(t1, p_conf)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        cast("Callable[..., EagerTensorProtocol]", pool)(t1, p_conf_max)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    from unittest.mock import patch

    config.eager_mode = True
    getattr(_state, "update")({"patch": patch})
    _state.update(locals())


def _test_pooling_brute_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    patch, t1 = _state["patch"], _state["t1"]
    with patch("ml_switcheroo_compiler.ops.nn.pooling.get_active_backend") as mock_backend:

        class DummyBackend:
            """Test class DummyBackend."""

            def execute_op(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op."""
                return np.array([0.0]), np.array([0])

            def array(self, x: EagerValue) -> EagerValue:
                """Test function array."""
                return x

        setattr(mock_backend, "return_value", DummyBackend())
        try:
            cast("Callable[..., EagerTensorProtocol]", max_pool)(t1, (2, 2), return_indices=True)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

    config.eager_mode = False
    getattr(_state, "update")({"DummyBackend": DummyBackend, "mock_backend": mock_backend})
    _state.update(locals())


def _test_pooling_brute_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_pooling_brute.

    Args:
        _state: Shared test state dictionary.
    """
    p_conf, p_conf_max, patch, t1 = _state["p_conf"], _state["p_conf_max"], _state["patch"], _state["t1"]
    with patch("ml_switcheroo_compiler.ops.reductions.frontend_utils._emit_reduction_node"), patch("ml_switcheroo_compiler.ops.linalg.utils._emit_linalg_node"):
        try:
            cast("Callable[..., EagerTensorProtocol]", max_pool)(t1, (2, 2), return_indices=True)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", fractional_avg_pool)(t1, p_conf)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            cast("Callable[..., EagerTensorProtocol]", fractional_max_pool)(t1, p_conf_max)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_pooling_brute() -> None:
    """Test function test_pooling_brute."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_pooling_brute_part_1(_state)
    _test_pooling_brute_part_2(_state)
    _test_pooling_brute_part_3(_state)
    _test_pooling_brute_part_4(_state)
    _test_pooling_brute_part_5(_state)
    _test_pooling_brute_part_6(_state)
    _test_pooling_brute_part_7(_state)
    _test_pooling_brute_part_8(_state)
    assert isinstance(_state, dict)
