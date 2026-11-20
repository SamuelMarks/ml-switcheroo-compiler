"""Tests for test_tensor_mixins_delegation."""

from __future__ import annotations

import sys
import typing

import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from tests.testing_utils import MockTestStub, coerce_int, ensure_tensor, np

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def make_tensor(data: BackendArg) -> Tensor:
    """Test function make_tensor.

    Args:
        data: Test parameter `data`.

    Returns:
        Tensor: Computed output value from `make_tensor`.
    """
    cfg = TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=())
    return Tensor(data, cfg)


def test_numpy_conversion_fallback() -> None:
    """Test function test_numpy_conversion_fallback."""
    from unittest.mock import patch

    t = make_tensor(1.0)
    with patch.dict(sys.modules, {"numpy": None}):
        with pytest.raises((ImportError, ModuleNotFoundError, RuntimeError)):
            ensure_tensor(t).numpy()


def test_exception_in_asarray() -> None:
    """Test function test_exception_in_asarray."""

    class BadData(MockTestStub):
        """Test class BadData."""

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1.0]

        def __array__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __array__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Raises:
                RuntimeError: When raised during `__array__` (boom).
            """
            raise RuntimeError("boom")

    t = make_tensor(BadData())
    res = np.asarray(t)
    assert res.shape == (1,)


def test_item_eval_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_item_eval_fallback.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """

    class FakeBackend:
        """Test class FakeBackend."""

        def item(self, x: BackendArg) -> BackendArg:
            """Test function item.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return coerce_int(x)

    class FakeTensor(Tensor):
        """Test class FakeTensor."""

        def eval(self) -> Tensor:
            """Test function eval.

            Returns:
                Tensor: Computed output value from `eval`.
            """
            return ensure_tensor(5)

    t = FakeTensor(1.0, TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=()))
    import ml_switcheroo_compiler.backends.registry as reg

    monkeypatch.setattr(reg, "get_active_backend", lambda: FakeBackend())
    assert t.item() == 5


def test_getitem_indexerror() -> None:
    """Test function test_getitem_indexerror."""
    t = make_tensor([1, 2])
    with pytest.raises(IndexError):
        ensure_ndarray(t)[(1, 2)]
    from ml_switcheroo_compiler.core import config

    orig = config.eager_mode
    config.eager_mode = False

    class FakeProxy:
        """Test class FakeProxy."""

        data = None

        def __index__(self) -> int:
            """Test function __index__.

            Returns:
                int: Unreachable return annotation.

            Raises:
                IndexError: When raised during `__index__` (only integers).
            """
            raise IndexError("only integers")

    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", lambda node: None, raising=False)
    try:
        res = t[FakeProxy()]
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", orig_add_node, raising=False)
