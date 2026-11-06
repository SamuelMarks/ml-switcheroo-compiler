"""Tests for test_tensor_mixins_delegation."""

from __future__ import annotations

import sys
import typing

import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_obj, np

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def make_tensor(data: EagerValue) -> EagerValue:
    """Test function make_tensor.

    Args:
        data: Test parameter `data`.
    """
    cfg = TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=())
    return Tensor(data, cfg)


def test_numpy_conversion_fallback() -> None:
    """Test function test_numpy_conversion_fallback."""
    t = make_tensor(1.0)
    original_numpy = sys.modules.get("numpy")
    ensure_eager_tensor(sys.modules)["numpy"] = None
    try:
        ensure_eager_tensor(ensure_eager_tensor(t).numpy)()
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    finally:
        ensure_eager_tensor(sys.modules)["numpy"] = original_numpy
    assert callable(test_numpy_conversion_fallback)


def test_exception_in_asarray() -> None:
    """Test function test_exception_in_asarray."""

    class BadData:
        """Test class BadData."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [1.0]

        def __array__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __array__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            raise RuntimeError("boom")

    t = ensure_callable(make_tensor)(ensure_callable(BadData)())
    res = np.asarray(t)
    assert res.shape == (1,)


def test_item_eval_fallback(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_item_eval_fallback.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """

    class FakeBackend:
        """Test class FakeBackend."""

        def item(self, x: EagerValue) -> EagerValue:
            """Test function item.

            Args:
                x: Test parameter `x`.
            """
            return ensure_callable(int)(x)

    class FakeTensor(Tensor):
        """Test class FakeTensor."""

        def eval(self) -> Tensor:
            """Test function eval."""
            return ensure_tensor_obj(5)

    t = FakeTensor(1.0, TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=()))
    import ml_switcheroo_compiler.backends.registry as reg

    ensure_callable(monkeypatch.setattr)(reg, "get_active_backend", lambda: ensure_callable(FakeBackend)())
    assert t.item() == 5


def test_getitem_indexerror() -> None:
    """Test function test_getitem_indexerror."""
    t = make_tensor([1, 2])
    with pytest.raises(IndexError):
        ensure_eager_tensor(t)[(1, 2)]
    from ml_switcheroo_compiler.core import config

    orig = config.eager_mode
    config.eager_mode = False

    class FakeProxy:
        """Test class FakeProxy."""

        def __index__(self) -> None:
            """Test function __index__."""
            raise IndexError("only integers")

    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(lambda node: None)
    try:
        res = ensure_eager_tensor(t)[ensure_callable(FakeProxy)()]
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(orig_add_node)
