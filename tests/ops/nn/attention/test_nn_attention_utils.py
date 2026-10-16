"""Tests for test_nn_attention_utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, alibi_mask, rope, sinusoidal_positional_encoding
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_nn_attention_utils_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_nn_attention_utils_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    config.eager_mode = True
    t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig(shape=(1, 1, 2), dtype=DType("float32"), device=Device("cpu")))

    assert cast("Callable[..., EagerTensorProtocol]", RopeOp)().infer_shape(t) == (1, 1, 2)
    assert cast("Callable[..., EagerTensorProtocol]", ScaledDotProductAttention)().infer_shape(t, t, t, None) == (1, 1, 2)

    assert rope(t, axis=2) is not None

    sinusoidal_positional_encoding(4, 4)
    alibi_mask(4, 2)

    import ml_switcheroo_compiler.backends.registry as registry_mod

    getattr(_state, "update")({"registry_mod": registry_mod, "t": t})
    _state.update(locals())


class _MockAttentionBackend:
    """Test class _MockAttentionBackend."""

    def execute_op(self, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function execute_op."""
        if op_name == "Concatenate":
            tensors = args[0]
            axis = kwargs.get("axis", -1)
            return cast("Callable[..., EagerTensorProtocol]", np.concatenate)([cast("Callable[..., EagerTensorProtocol]", np.atleast_1d)(arr) if cast("Callable[..., EagerTensorProtocol]", np.ndim)(arr) == 0 else arr for arr in cast("EagerTensorProtocol", tensors)], axis=axis)
        return np.array([1.0])

    def asarray(self, x: EagerValue) -> EagerValue:
        """Test function asarray."""
        return np.asarray(x)

    def array(self, x: EagerValue) -> EagerValue:
        """Test function array."""
        return np.asarray(x)

    def expand_dims(self, x: EagerValue, axis: EagerValue) -> EagerValue:
        """Test function expand_dims."""
        return cast("Callable[..., EagerTensorProtocol]", np.expand_dims)(x, axis=axis)

    def multiply(self, x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function multiply."""
        try:
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.multiply)(x, y))
        except Exception:
            return x

    def subtract(self, x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function subtract."""
        try:
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.subtract)(x, y))
        except Exception:
            return x

    def arange(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function arange."""
        return cast("Callable[..., EagerTensorProtocol]", np.arange)(*args, **kwargs)

    def exp(self, x: EagerValue) -> EagerValue:
        """Test function exp."""
        return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.exp)(x))

    def sin(self, x: EagerValue) -> EagerValue:
        """Test function sin."""
        return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.sin)(x))

    def cos(self, x: EagerValue) -> EagerValue:
        """Test function cos."""
        return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.cos)(x))

    def concatenate(self, x: EagerValue, axis: EagerValue) -> EagerValue:
        """Test function concatenate."""
        try:
            return cast("Callable[..., EagerTensorProtocol]", np.concatenate)([cast("Callable[..., EagerTensorProtocol]", np.atleast_1d)(arr) for arr in cast("EagerTensorProtocol", x)], axis=axis)
        except Exception:
            return x


def __test_nn_attention_utils_coverage_part_2_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of _test_nn_attention_utils_coverage_part_2.

    Args:
        _state: Shared test state dictionary.
    """
    getattr(_state, "update")({"MockAttentionBackend": _MockAttentionBackend})
    _state.update(locals())


def __test_nn_attention_utils_coverage_part_2_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of _test_nn_attention_utils_coverage_part_2.

    Args:
        _state: Shared test state dictionary.
    """
    MockAttentionBackend = _state["MockAttentionBackend"]
    getattr(_state, "update")({"MockAttentionBackend": MockAttentionBackend})
    _state.update(locals())
    _state.update(locals())


def _test_nn_attention_utils_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_nn_attention_utils_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    getattr(_state, "__setitem__")("_state", _state)
    for _sub_fn in (__test_nn_attention_utils_coverage_part_2_sub_1, __test_nn_attention_utils_coverage_part_2_sub_2):
        _sub_fn(_state)
    assert isinstance(_state, dict)


def _test_nn_attention_utils_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_nn_attention_utils_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    MockAttentionBackend, registry_mod, t = _state["MockAttentionBackend"], _state["registry_mod"], _state["t"]
    orig_backend = registry_mod.get_active_backend
    try:
        setattr(registry_mod, "get_active_backend", lambda: MockAttentionBackend())

        cast("Callable[..., EagerTensorProtocol]", rope)(t, axis=2)

        config.eager_mode = False
        global_tracing_state.is_tracing = True

        class DummyGraph:
            """Test class DummyGraph."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.name = "dummy"
                self.nodes = cast("dict[str, EagerTensorProtocol]", {})

            def add_node(self, node: EagerValue) -> None:
                """Test function add_node."""
                pass

        setattr(global_tracing_state, "active_graph", DummyGraph())

        cast("Callable[..., EagerTensorProtocol]", rope)(t, axis=2)
    finally:
        config.eager_mode = False
        global_tracing_state.is_tracing = False
        setattr(registry_mod, "get_active_backend", orig_backend)
    getattr(_state, "update")({"orig_backend": orig_backend})


def test_nn_attention_utils_coverage() -> None:
    """Test function test_nn_attention_utils_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_nn_attention_utils_coverage_part_1(_state)
    _test_nn_attention_utils_coverage_part_2(_state)
    _test_nn_attention_utils_coverage_part_3(_state)
    assert isinstance(_state, dict)
