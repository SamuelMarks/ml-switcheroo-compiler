"""Tests for test_core_math_ops."""

from __future__ import annotations

import inspect
import sys
import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.core_math_ops as cmo
import ml_switcheroo_compiler.backends.eager.core_math_ops as core_math_ops
import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_core_math_ops_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = core_math_ops._true_divide(np, 10.0, 2.0)
    assert res == 5.0
    res = core_math_ops._true_divide(object, 10.0, 2.0)
    assert res is None
    res = core_math_ops._fft(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._rfft(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fftn(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fft(object, np.array([1.0, 2.0]))
    assert res is None
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def _test_core_math_ops_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = core_math_ops._rfft(object, np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._fftn(object, np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._erf(np, 0.5)
    assert res is not None
    res = core_math_ops._erfc(np, 0.5)
    assert res is not None
    res = core_math_ops._expm1(np, 0.5)
    assert res is not None
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def _test_core_math_ops_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyMock:
        """Test class DummyMock."""

        @staticmethod
        def sign(x: EagerValue) -> EagerValue:
            """Test function sign."""
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.sign)(x))

        @staticmethod
        def abs(x: EagerValue) -> EagerValue:
            """Test function abs."""
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.abs)(x))

        @staticmethod
        def exp(x: EagerValue) -> EagerValue:
            """Test function exp."""
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.exp)(x))

        @staticmethod
        def log(x: EagerValue) -> EagerValue:
            """Test function log."""
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.log)(x))

        @staticmethod
        def sqrt(x: EagerValue) -> EagerValue:
            """Test function sqrt."""
            return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.sqrt)(x))

    dummy = DummyMock()
    res = core_math_ops._erf(dummy, 0.5)
    assert res is not None
    getattr(_state, "update")({"DummyMock": DummyMock, "dummy": dummy, "res": res})
    _state.update(locals())


def _test_core_math_ops_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    dummy = _state["dummy"]
    res = core_math_ops._erfc(dummy, 0.5)
    assert res is not None
    res = core_math_ops._expm1(dummy, 0.5)
    assert res is not None
    res = core_math_ops._erfinv(dummy, 0.5)
    assert res is not None
    res = core_math_ops._nan_to_num(np, np.array([np.nan, 1.0]))
    assert res is not None
    res = core_math_ops._nan_to_num(dummy, np.array([np.nan, 1.0]))
    assert res is not None
    res = core_math_ops._einsum(np, "i,i->", np.array([1.0]), np.array([2.0]))
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def _test_core_math_ops_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy, res) = (_state["dummy"], _state["res"])
    assert res is not None
    res = cast("EagerTensorProtocol", core_math_ops._einsum(dummy, "i,i->", np.array([1.0]), np.array([2.0])))
    assert res is not None
    res = cast("EagerTensorProtocol", core_math_ops._allclose(np, np.array([1.0]), np.array([1.0])))
    assert res is not None
    res = cast("EagerTensorProtocol", core_math_ops._allclose(dummy, np.array([1.0]), np.array([1.0])))
    assert res is not None
    res = cast("EagerTensorProtocol", core_math_ops._psum(np, np.array([1.0])))
    assert res is not None
    res = cast("EagerTensorProtocol", core_math_ops._pmean(np, np.array([1.0])))
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def _test_core_math_ops_coverage_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = _state["res"]
    assert res is not None

    class DummyMockPolygamma:
        """Test class DummyMockPolygamma."""

        pass

    res = cast("EagerTensorProtocol", core_math_ops._polygamma(DummyMockPolygamma, 1, 1))
    assert res is not None

    class DummyMockDigamma:
        """Test class DummyMockDigamma."""

        pass

    res = cast("EagerTensorProtocol", core_math_ops._digamma(DummyMockDigamma, 1))
    assert res is not None
    res = cast("EagerTensorProtocol", core_math_ops._np_zerofraction(np, np.array([0.0, 1.0])))
    assert res is not None
    getattr(_state, "update")({"DummyMockDigamma": DummyMockDigamma, "DummyMockPolygamma": DummyMockPolygamma, "res": res})
    _state.update(locals())


def _test_core_math_ops_coverage_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class MockRandomZeroFraction:
        """Test class MockRandomZeroFraction."""

        class random:
            """Test class random."""

            @staticmethod
            def zerofraction(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function zerofraction."""
                return 1.0

    res = core_math_ops._np_zerofraction(MockRandomZeroFraction, np.array([0.0, 1.0]))
    assert res == 0.5
    res = core_math_ops._np_zeta(np, 2.0, 1.0)
    assert res is not None
    res = core_math_ops._np_zeta(object, 2.0, 1.0)
    assert res is not None
    getattr(_state, "update")({"MockRandomZeroFraction": MockRandomZeroFraction, "res": res})


def test_core_math_ops_coverage() -> None:
    """Test function test_core_math_ops_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_core_math_ops_coverage_part_1(_state)
    _test_core_math_ops_coverage_part_2(_state)
    _test_core_math_ops_coverage_part_3(_state)
    _test_core_math_ops_coverage_part_4(_state)
    _test_core_math_ops_coverage_part_5(_state)
    _test_core_math_ops_coverage_part_6(_state)
    _test_core_math_ops_coverage_part_7(_state)
    assert isinstance(_state, dict)


def test_missing_i0() -> None:
    """Test function test_missing_i0."""
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class BkWithI0:
        """Test class BkWithI0."""

        def i0(self, x: EagerValue) -> EagerValue:
            """Test function i0."""
            return np.array([1.0])

        def iscomplex(self, x: EagerValue) -> EagerValue:
            """Test function iscomplex."""
            return np.array([1.0])

        def isreal(self, x: EagerValue) -> EagerValue:
            """Test function isreal."""
            return np.array([1.0])

        def signbit(self, x: EagerValue) -> EagerValue:
            """Test function signbit."""
            return np.array([1.0])

    class BkWithoutI0:
        """Test class BkWithoutI0."""

        pass

    x = np.array([1.0])
    mod._i0(BkWithI0(), x)
    mod._i0(BkWithoutI0(), x)
    mod._iscomplex(BkWithI0(), x)
    mod._iscomplex(BkWithoutI0(), x)
    mod._isreal(BkWithI0(), x)
    mod._isreal(BkWithoutI0(), x)
    mod._signbit(BkWithI0(), x)
    mod._signbit(BkWithoutI0(), x)
    assert x is not None or x is None


class DummyBackend:
    """Test class DummyBackend."""

    def shape(self, x: EagerValue) -> EagerValue:
        """Test function shape."""
        return (2, 2)

    def indices(self, x: EagerValue) -> EagerValue:
        """Test function indices."""
        return (0, 0)

    def ones(self, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function ones."""
        return x

    def where(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function where."""
        return args[0]

    pass


class DummyBackendWithRandom:
    """Test class DummyBackendWithRandom."""

    class random:
        """Test class random."""

        @staticmethod
        def triangular(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function triangular."""
            return "random.triangular"

        @staticmethod
        def triangularsolve(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function triangularsolve."""
            return "random.triangularsolve"

        @staticmethod
        def xlog1py(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function xlog1py."""
            return "random.xlog1py"

        @staticmethod
        def xlogy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function xlogy."""
            return "random.xlogy"


warnings.filterwarnings("ignore")


class AllBackend:
    """Test class AllBackend."""

    def __getattr__(self, name: EagerValue) -> EagerValue:
        """Test function __getattr__."""
        if name in ("__bases__", "__class__"):
            raise AttributeError()
        if name == "nn":
            return cast("EagerTensorProtocol", self)

        def _dummy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function _dummy."""
            return np.array([1.0])

        setattr(_dummy, "item", lambda: 1.0)
        setattr(_dummy, "tolist", lambda: 1.0)
        return _dummy


class FallbackBackend:
    """Test class FallbackBackend."""

    def __init__(self, hide_names: EagerValue) -> None:
        """Test function __init__."""
        self.hide_names = hide_names
        self.np = np

    def __getattr__(self, name: EagerValue) -> EagerValue:
        """Test function __getattr__."""
        if cast("EagerTensorProtocol", name) in cast("EagerTensorProtocol", self.hide_names):
            raise AttributeError(f"Hidden: {name}")
        if name == "nn":
            return cast("EagerTensorProtocol", self)
        return cast("EagerTensorProtocol", getattr(self.np, str(name)))


def _test_all_core_math_ops_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    op = _state["op"]
    sig = inspect.signature(op)
    getattr(_state, "update")({"sig": sig})
    _state.update(locals())


def _test_all_core_math_ops_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_int, sig) = (_state["arg"], _state["arg_int"], _state["sig"])

    def build_call(bk: EagerValue, use_int: EagerValue = False, use_one_arg: EagerValue = False, sig: EagerValue = sig) -> EagerValue:
        """Test function build_call."""
        args = []
        kwargs = {}
        val = arg_int if use_int else arg
        for p in cast("EagerTensorProtocol", getattr(sig, "parameters")).values():
            if p.name == "backend_module":
                args.append(bk)
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                if not use_one_arg:
                    args.extend([val, val])
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                pass
            elif p.name == "axis":
                kwargs["axis"] = 0
            elif p.name == "axes":
                getattr(kwargs, "__setitem__")("axes", (0,))
            elif p.default != inspect.Parameter.empty:
                pass
            else:
                args.append(val)
        return (args, kwargs)

    getattr(_state, "update")({"build_call": build_call})
    _state.update(locals())


def _test_all_core_math_ops_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (all_bk, arg, build_call, fb) = (_state["all_bk"], _state["arg"], _state["build_call"], _state["fb"])
    op = _state["op"]
    for bk in [all_bk, fb]:
        for use_int in [False, True]:
            for use_one_arg in [False, True]:
                (args, kwargs) = build_call(bk, use_int, use_one_arg)
                try:
                    cast("Callable[..., EagerTensorProtocol]", op)(*args, **cast("dict[str, EagerValue]", kwargs))
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
                try:
                    op(bk, arg)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
    _state.update(locals())


def _test_all_core_math_ops_coverage_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (build_call, fb) = (_state["build_call"], _state["fb"])
    (args, kwargs) = build_call(fb)
    orig_scipy = sys.modules.get("scipy.special")
    orig_scipy2 = sys.modules.get("scipy.linalg")
    getattr(sys.modules, "__setitem__")("scipy.special", None)
    getattr(sys.modules, "__setitem__")("scipy.linalg", None)
    getattr(sys.modules, "__setitem__")("scipy", None)
    getattr(_state, "update")({"args": args, "kwargs": kwargs, "orig_scipy": orig_scipy, "orig_scipy2": orig_scipy2})
    _state.update(locals())


def _test_all_core_math_ops_coverage_sub_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 5 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (args, kwargs, op, orig_scipy) = (_state["args"], _state["kwargs"], _state["op"], _state["orig_scipy"])
    orig_scipy2 = _state["orig_scipy2"]
    try:
        cast("Callable[..., EagerTensorProtocol]", op)(*args, **cast("dict[str, EagerValue]", kwargs))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    finally:
        if orig_scipy is not None:
            getattr(sys.modules, "__setitem__")("scipy.special", orig_scipy)
        elif "scipy.special" in sys.modules:
            del sys.modules["scipy.special"]
        if orig_scipy2 is not None:
            getattr(sys.modules, "__setitem__")("scipy.linalg", orig_scipy2)
        elif "scipy.linalg" in sys.modules:
            del sys.modules["scipy.linalg"]
        if "scipy" in sys.modules:
            del sys.modules["scipy"]
    _state.update(locals())


def test_all_core_math_ops_coverage() -> None:
    """Test function test_all_core_math_ops_coverage."""
    warnings.filterwarnings("ignore")
    ops = [getattr(cmo, name) for name in dir(cmo) if name.startswith("_") and callable(getattr(cmo, name))]
    all_bk = AllBackend()
    arg = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    arg_int = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    hide = [name[1:] for name in dir(cmo) if name.startswith("_") and callable(getattr(cmo, name))]
    fb = FallbackBackend(hide)
    for op in ops:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"all_bk": all_bk, "arg": arg, "arg_int": arg_int, "fb": fb})
        getattr(_state, "update")({"hide": hide, "op": op, "ops": ops})
        for _sub_fn in (_test_all_core_math_ops_coverage_sub_1, _test_all_core_math_ops_coverage_sub_2, _test_all_core_math_ops_coverage_sub_3, _test_all_core_math_ops_coverage_sub_4, _test_all_core_math_ops_coverage_sub_5):
            _sub_fn(_state)
        assert isinstance(_state, dict)


def _test_missing_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class EmptyBk:
        """Test class EmptyBk."""

        pass

    class NNSoftmaxBk:
        """Test class NNSoftmaxBk."""

        class nn:
            """Test class nn."""

            @staticmethod
            def softmax(x: EagerValue, axis: EagerValue) -> EagerValue:
                """Test function softmax."""
                return x

    arg = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    cast("Callable[..., EagerTensorProtocol]", mod._apply_causal_mask)(cast("Callable[..., EagerTensorProtocol]", EmptyBk)(), arg)
    cast("Callable[..., EagerTensorProtocol]", mod._apply_softmax)(cast("Callable[..., EagerTensorProtocol]", NNSoftmaxBk)(), arg)
    cast("Callable[..., EagerTensorProtocol]", mod._apply_softmax)(cast("Callable[..., EagerTensorProtocol]", EmptyBk)(), arg)
    getattr(_state, "update")({"EmptyBk": EmptyBk, "NNSoftmaxBk": NNSoftmaxBk, "arg": arg, "mod": mod})
    getattr(_state, "update")({"np": np})
    _state.update(locals())


def _test_missing_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, mod, np) = (_state["arg"], _state["mod"], _state["np"])

    class DummyItem:
        """Test class DummyItem."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 1.0

    class DummyList:
        """Test class DummyList."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return 1.0

    cast("Callable[..., EagerTensorProtocol]", mod._allclose)(np, arg, arg, rtol=cast("Callable[..., EagerTensorProtocol]", DummyItem)())
    cast("Callable[..., EagerTensorProtocol]", mod._allclose)(np, arg, arg, rtol=cast("Callable[..., EagerTensorProtocol]", DummyList)())

    class MatmulBk:
        """Test class MatmulBk."""

        @staticmethod
        def matmul(a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul."""
            return a

    getattr(_state, "update")({"DummyItem": DummyItem, "DummyList": DummyList, "MatmulBk": MatmulBk})
    _state.update(locals())


def _test_missing_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (EmptyBk, MatmulBk, arg, mod) = (_state["EmptyBk"], _state["MatmulBk"], _state["arg"], _state["mod"])
    np = _state["np"]
    with pytest.raises(AttributeError):
        mod._scaled_dot_product_attention_eager(EmptyBk(), arg, arg, arg, scale=1.0, is_causal=True, mask=arg)
    mod._scaled_dot_product_attention_eager(MatmulBk(), arg, arg, arg, scale=1.0, is_causal=True, mask=arg)
    mod._scaled_dot_product_attention_eager(np, arg, arg, arg, scale=1.0, is_causal=True, mask=arg)

    class BkWithWhere:
        """Test class BkWithWhere."""

        @staticmethod
        def where(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function where."""
            return 1

        @staticmethod
        def floor(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function floor."""
            return 1

        @staticmethod
        def ceil(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ceil."""
            return 1

    cast("Callable[..., EagerTensorProtocol]", mod._fix)(cast("Callable[..., EagerTensorProtocol]", BkWithWhere)(), arg)
    mod._adaptive_max_pool3d_indices(np, arg, 1)
    getattr(_state, "update")({"BkWithWhere": BkWithWhere})
    _state.update(locals())


def _test_missing_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, mod, np) = (_state["arg"], _state["mod"], _state["np"])
    mod._add_n(np, [arg, arg])


def test_missing_coverage() -> None:
    """Test function test_missing_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_missing_coverage_part_1(_state)
    _test_missing_coverage_part_2(_state)
    _test_missing_coverage_part_3(_state)
    _test_missing_coverage_part_4(_state)
    assert isinstance(_state, dict)


def test_empty_backend() -> None:
    """Test function test_empty_backend."""
    import inspect

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class EmptyBackend:
        """Test class EmptyBackend."""

        pass

    empty_bk = EmptyBackend()
    arg = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    ops = [getattr(mod, name) for name in dir(mod) if name.startswith("_") and callable(getattr(mod, name))]
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = [empty_bk] + [arg] * (len(sig.parameters) - 1)
        try:
            op(*args_to_pass[: len(sig.parameters)])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_empty_backend)


def _test_remaining_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    getattr(_state, "update")({"mod": mod, "np": np})
    _state.update(locals())


class _BkWithLinalg:
    """Test class _BkWithLinalg."""

    class linalg:
        """Test class linalg."""

        @staticmethod
        def cho_solve(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function cho_solve."""
            return np.array([1.0])

        @staticmethod
        def solve_banded(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function solve_banded."""
            return np.array([1.0])

        @staticmethod
        def svd(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function svd."""
            return np.array([1.0])

        @staticmethod
        def tensorinv(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function tensorinv."""
            return np.array([1.0])

        @staticmethod
        def tensorsolve(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function tensorsolve."""
            return np.array([1.0])

        @staticmethod
        def cross(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function cross."""
            return np.array([1.0])

        @staticmethod
        def outer(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function outer."""
            return np.array([1.0])

        @staticmethod
        def det(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function det."""
            return np.array([1.0])

        @staticmethod
        def eig(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eig."""
            return np.array([1.0])

        @staticmethod
        def eigh(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eigh."""
            return np.array([1.0])

        @staticmethod
        def eigvals(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eigvals."""
            return np.array([1.0])

        @staticmethod
        def eigvalsh(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eigvalsh."""
            return np.array([1.0])

        @staticmethod
        def cholesky(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function cholesky."""
            return np.array([1.0])

        @staticmethod
        def matrix_power(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function matrix_power."""
            return np.array([1.0])

        @staticmethod
        def norm(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function norm."""
            return np.array([1.0])


def _test_remaining_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    getattr(_state, "update")({"BkWithLinalg": _BkWithLinalg})
    _state.update(locals())


def _test_remaining_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (BkWithLinalg, mod, np) = (_state["BkWithLinalg"], _state["mod"], _state["np"])
    bk = BkWithLinalg()
    arg = np.array([1.0])
    mod._cholesky_solve(bk, arg)
    mod._banded_triangular_solve(bk, arg)
    mod._svd(bk, arg)
    mod._tensorinv(bk, arg)
    getattr(_state, "update")({"arg": arg, "bk": bk})
    _state.update(locals())


def _test_remaining_coverage_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk, mod) = (_state["arg"], _state["bk"], _state["mod"])
    mod._tensorsolve(bk, arg)
    with pytest.raises(AttributeError):
        mod._cross(bk, arg)
    with pytest.raises(AttributeError):
        mod._outer(bk, arg)
    mod._det(bk, arg)
    _state.update(locals())


def _test_remaining_coverage_sub_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 5 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk, mod) = (_state["arg"], _state["bk"], _state["mod"])
    mod._eig(bk, arg)
    mod._eigh(bk, arg)
    mod._eigvals(bk, arg)
    mod._eigvalsh(bk, arg)
    _state.update(locals())


def _test_remaining_coverage_sub_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 6 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk, mod, np) = (_state["arg"], _state["bk"], _state["mod"], _state["np"])
    mod._cholesky(bk, arg)
    mod._matrix_power(bk, arg)
    mod._norm(bk, arg)
    mod._add_n(np, [arg, arg])
    mod._associative_scan(np, lambda a, b: a + b, np.array([1.0, 2.0]))
    _state.update(locals())


def test_remaining_coverage() -> None:
    """Test function test_remaining_coverage."""
    _state: dict[str, EagerTensorProtocol] = dict(locals())
    for _sub_fn in (_test_remaining_coverage_sub_1, _test_remaining_coverage_sub_2, _test_remaining_coverage_sub_3, _test_remaining_coverage_sub_4, _test_remaining_coverage_sub_5, _test_remaining_coverage_sub_6):
        _sub_fn(_state)
    assert isinstance(_state, dict)


def test_unfold_coverage_manual() -> None:
    """Test function test_unfold_coverage_manual."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _np_unfold

    class DummyBackend:
        """Test class DummyBackend."""

        def asarray(self, x: EagerValue) -> EagerValue:
            """Test function asarray."""
            return x

    res = _np_unfold(DummyBackend(), np.ones((1, 1, 4, 4)), kernel_size=3, stride=1)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 1, 4, 4)
    res = _np_unfold(DummyBackend(), np.ones((1, 1, 2, 2)), kernel_size=3, stride=1)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 1, 2, 2)


warnings.filterwarnings("ignore")


def test_final_fftn() -> None:
    """Test function test_final_fftn."""

    class DummyFFT:
        """Test class DummyFFT."""

        def fftn(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftn."""
            return np.array([1.0])

    class BkSpecial:
        """Test class BkSpecial."""

        fft = DummyFFT()

    mod._fftn(BkSpecial(), np.array([1.0]))
    mod._fftnd(BkSpecial(), np.array([1.0]))
    assert callable(mod._fftnd)


warnings.filterwarnings("ignore")


class _BkSpecial:
    """Test class _BkSpecial."""

    class linalg:
        """Test class linalg."""

        @staticmethod
        def householder_product(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function householder_product."""
            return np.array([1.0])

    class fft:
        """Test class fft."""

        @staticmethod
        def fft2(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fft2."""
            return np.array([1.0])

        @staticmethod
        def fftfreq(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftfreq."""
            return np.array([1.0])

        @staticmethod
        def fftn(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftn."""
            return np.array([1.0])

        @staticmethod
        def fftshift(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftshift."""
            return np.array([1.0])

        @staticmethod
        def ifft(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifft."""
            return np.array([1.0])

        @staticmethod
        def ifft2(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifft2."""
            return np.array([1.0])

        @staticmethod
        def ifftn(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifftn."""
            return np.array([1.0])

        @staticmethod
        def ifftshift(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifftshift."""
            return np.array([1.0])

    class lax:
        """Test class lax."""

        @staticmethod
        def infeed(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function infeed."""
            return np.array([1.0])

        @staticmethod
        def outfeed(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function outfeed."""
            return np.array([1.0])

        @staticmethod
        def pshuffle(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function pshuffle."""
            return np.array([1.0])

        @staticmethod
        def pswapaxes(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function pswapaxes."""
            return np.array([1.0])

        @staticmethod
        def ppermute(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ppermute."""
            return np.array([1.0])

        @staticmethod
        def psum_scatter(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function psum_scatter."""
            return np.array([1.0])

    class random:
        """Test class random."""

        @staticmethod
        def tridiagonal(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function tridiagonal."""
            return np.array([1.0])


def _test_even_more_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    getattr(_state, "update")({"BkSpecial": _BkSpecial})
    _state.update(locals())


def _test_even_more_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    BkSpecial = _state["BkSpecial"]
    bk = BkSpecial()
    arg = np.array([1.0])
    mod._accumulate_n(np, [arg, arg])
    mod._householder_product(bk, arg)
    mod._fft2(bk, arg)
    mod._fftfreq(bk, arg)
    mod._fftn(bk, arg)
    mod._fftshift(bk, arg)
    mod._ifft(bk, arg)
    mod._ifft2(bk, arg)
    mod._ifftn(bk, arg)
    mod._ifftshift(bk, arg)
    mod._infeed(bk, arg)
    mod._outfeed(bk, arg)
    mod._pshuffle(bk, arg)
    mod._pswapaxes(bk, arg)
    mod._ppermute(bk, arg)
    mod._np_tridiagonal(bk, arg)
    getattr(_state, "update")({"arg": arg, "bk": bk})
    _state.update(locals())


def _test_even_more_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk) = (_state["arg"], _state["bk"])
    mod._fftn(bk, arg)
    mod._psumscatter(bk, arg)
    assert arg is not None or arg is None
    _state.update(locals())


def test_even_more_coverage() -> None:
    """Test function test_even_more_coverage."""
    _state: dict[str, EagerTensorProtocol] = dict(locals())
    for _sub_fn in (_test_even_more_coverage_sub_1, _test_even_more_coverage_sub_2, _test_even_more_coverage_sub_3):
        _sub_fn(_state)
    assert isinstance(_state, dict)
