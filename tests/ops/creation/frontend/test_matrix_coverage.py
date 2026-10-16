"""Tests for test_matrix_coverage."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.creation.frontend_matrix as matrix_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def _test_frontend_matrix_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_frontend_matrix_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    config.eager_mode = True
    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.eye(3, dtype=np.float32)
    getattr(_state, "update")({"mock_backend": mock_backend})
    _state.update(locals())


def _test_frontend_matrix_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_frontend_matrix_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    mock_backend = _state["mock_backend"]
    with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix.get_active_backend", return_value=mock_backend):
        e1 = matrix_mod.eye(3)
        assert cast("EagerTensorProtocol", getattr(e1, "shape")) == (3, 3)
        e2 = matrix_mod.eye(3, m=4, k=1, dtype=DType.Float64, device=Device("cpu"))
        assert cast("EagerTensorProtocol", getattr(e2, "shape")) == (3, 4)

        # custom dtype without .value
        class DummyDtypeWithoutValue:
            """Mock dtype without value attribute."""

            name: str = "float32"

        e3 = cast("Callable[..., EagerTensorProtocol]", matrix_mod.eye)(2, dtype=cast("Callable[..., EagerTensorProtocol]", DummyDtypeWithoutValue)())
        assert cast("EagerTensorProtocol", getattr(e3, "shape")) == (2, 2)

        # identity
        i1 = matrix_mod.identity(4)
        assert cast("EagerTensorProtocol", getattr(i1, "shape")) == (4, 4)

        # _diag_eager with data having shape and raw dtype
        raw_input = np.array([[1.0, 2.0], [3.0, 4.0]])
        d1 = cast("Callable[..., EagerTensorProtocol]", matrix_mod._diag_eager)(raw_input, 0, cast("Callable[..., EagerTensorProtocol]", Device)("cpu"), None)
        assert isinstance(d1, Tensor)

        # _diag_eager with tensor input and data lacking shape
        mock_data_no_shape = 123.0
        setattr(mock_backend.execute_op, "return_value", mock_data_no_shape)
        d2 = matrix_mod._diag_eager(Tensor(raw_input, TensorConfig((2, 2), DType.Float32, Device("cpu"))), 0, Device("cpu"), DType.Float32)
        assert cast("EagerTensorProtocol", getattr(d2, "shape")) == ()

        # diag in eager mode
        setattr(mock_backend.execute_op, "return_value", np.array([1.0, 4.0]))
        t_matrix = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), DType.Float32, Device("cpu")))
        res_diag_eager = matrix_mod.diag(t_matrix, 0)
        assert isinstance(res_diag_eager, Tensor)
    getattr(_state, "update")({"DummyDtypeWithoutValue": DummyDtypeWithoutValue, "d1": d1, "d2": d2, "e1": e1})
    getattr(_state, "update")({"e2": e2, "e3": e3, "i1": i1, "mock_data_no_shape": mock_data_no_shape})
    getattr(_state, "update")({"raw_input": raw_input, "res_diag_eager": res_diag_eager, "t_matrix": t_matrix})
    _state.update(locals())


def _test_frontend_matrix_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_frontend_matrix_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    t_matrix = _state["t_matrix"]
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix._emit_creation_node", return_value="emitted_eye"):
        res_graph_eye = matrix_mod.eye(3, 3)
        assert res_graph_eye == "emitted_eye"

    # diag non-eager mode
    # 1. invalid shape length (!= 2)
    t_1d = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
    with pytest.raises(ValueError, match="diag requires a 1D or 2D tensor."):
        matrix_mod.diag(t_1d)

    # 2. not tracing
    with patch.object(global_tracing_state, "is_tracing", False):
        with pytest.raises(RuntimeError, match="Cannot emit diag node outside of a tracing context."):
            cast("Callable[..., EagerTensorProtocol]", matrix_mod.diag)(t_matrix)
    getattr(_state, "update")({"res_graph_eye": res_graph_eye, "t_1d": t_1d})
    _state.update(locals())


def _test_frontend_matrix_coverage_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_frontend_matrix_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    with patch.object(global_tracing_state, "is_tracing", True):
        with patch.object(global_tracing_state, "add_node") as mock_add_node:
            # With input.data having id and dtype having value
            class MockDataWithId:
                """Mock data payload with id."""

                id: str = "tensor_input_id"

            t_traced = cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("Callable[..., EagerTensorProtocol]", MockDataWithId)(), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((3, 3), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))
            res_traced = cast("Callable[..., EagerTensorProtocol]", matrix_mod.diag)(t_traced, 1)
            assert cast("EagerTensorProtocol", getattr(res_traced, "shape")) == (2,)
            mock_add_node.assert_called_once()

            # With input having id directly and dtype lacking value
            class MockInputDirectId:
                """Mock tensor object with id and custom dtype."""

                id: str = "direct_id"
                shape: tuple[int, ...] = (4, 4)
                dtype: str = "float32"
                device: None = None

            mock_add_node.reset_mock()
            res_traced2 = cast("Callable[..., EagerTensorProtocol]", matrix_mod.diag)(cast("Callable[..., EagerTensorProtocol]", MockInputDirectId)(), 0)
            assert cast("EagerTensorProtocol", getattr(res_traced2, "shape")) == (4,)
            mock_add_node.assert_called_once()
    getattr(_state, "update")({"MockDataWithId": MockDataWithId, "MockInputDirectId": MockInputDirectId, "mock_add_node": mock_add_node, "res_traced": res_traced})
    getattr(_state, "update")({"res_traced2": res_traced2, "t_traced": t_traced})
    _state.update(locals())


def test_frontend_matrix_coverage() -> None:
    """Verify 100% line and branch coverage for frontend_matrix operations."""
    orig_eager = config.eager_mode
    try:
        # eye & identity eager mode
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"orig_eager": orig_eager})
        for _sub_fn in (
            _test_frontend_matrix_coverage_sub_1,
            _test_frontend_matrix_coverage_sub_2,
            _test_frontend_matrix_coverage_sub_3,
            _test_frontend_matrix_coverage_sub_4,
        ):
            _sub_fn(_state)
        assert isinstance(_state, dict)
    finally:
        config.eager_mode = orig_eager
