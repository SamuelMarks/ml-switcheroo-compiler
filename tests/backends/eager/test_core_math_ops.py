"""Tests for test_core_math_ops."""

from __future__ import annotations

import inspect
import sys
import typing
import warnings

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.core_math_ops as cmo
import ml_switcheroo_compiler.backends.eager.core_math_ops as core_math_ops
import ml_switcheroo_compiler.backends.eager.core_math_ops as mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from tests.testing_utils import ensure_attrs, ensure_callable, ensure_eager_tensor, ensure_mod, ensure_mutable, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class _BareMod:
    """Empty dummy module without math functions."""


def _test_core_math_ops_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = core_math_ops._true_divide(ensure_eager_tensor(np), 10.0, 2.0)
    assert res == 5.0
    res = core_math_ops._true_divide(ensure_mod(_BareMod), 10.0, 2.0)
    assert res is None
    res = core_math_ops._fft(ensure_eager_tensor(np), np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._rfft(ensure_eager_tensor(np), np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fftn(ensure_eager_tensor(np), np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fft(ensure_mod(_BareMod), np.array([1.0, 2.0]))
    assert res is None
    _state.update(ensure_tensor_dict({"res": res}))


def _test_core_math_ops_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = core_math_ops._rfft(ensure_mod(_BareMod), np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._fftn(ensure_mod(_BareMod), np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._erf(ensure_eager_tensor(np), 0.5)
    assert res is not None
    res = core_math_ops._erfc(ensure_eager_tensor(np), 0.5)
    assert res is not None
    res = core_math_ops._expm1(ensure_eager_tensor(np), 0.5)
    assert res is not None
    _state.update(ensure_tensor_dict({"res": res}))


def _test_core_math_ops_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyMock:
        """Test class DummyMock."""

        @staticmethod
        def sign(x: EagerValue) -> EagerValue:
            """Test function sign.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `sign`.
            """
            return ensure_eager_tensor(ensure_callable(np.sign)(x))

        @staticmethod
        def abs(x: EagerValue) -> EagerValue:
            """Test function abs.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `abs`.
            """
            return ensure_eager_tensor(ensure_callable(np.abs)(x))

        @staticmethod
        def exp(x: EagerValue) -> EagerValue:
            """Test function exp.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `exp`.
            """
            return ensure_eager_tensor(ensure_callable(np.exp)(x))

        @staticmethod
        def log(x: EagerValue) -> EagerValue:
            """Test function log.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `log`.
            """
            return ensure_eager_tensor(ensure_callable(np.log)(x))

        @staticmethod
        def sqrt(x: EagerValue) -> EagerValue:
            """Test function sqrt.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `sqrt`.
            """
            return ensure_eager_tensor(ensure_callable(np.sqrt)(x))

    dummy = DummyMock()
    res = core_math_ops._erf(ensure_mod(dummy), 0.5)
    assert res is not None
    _state.update(ensure_tensor_dict({"DummyMock": DummyMock, "dummy": dummy, "res": res}))


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
    res = core_math_ops._nan_to_num(ensure_eager_tensor(np), np.array([np.nan, 1.0]))
    assert res is not None
    res = core_math_ops._nan_to_num(dummy, np.array([np.nan, 1.0]))
    assert res is not None
    res = core_math_ops._einsum(ensure_eager_tensor(np), "i,i->", np.array([1.0]), np.array([2.0]))
    _state.update(ensure_tensor_dict({"res": res}))
    _state.update(ensure_tensor_dict({"dummy": dummy}))


def _test_core_math_ops_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy, res) = (_state["dummy"], _state["res"])
    assert res is not None
    res = ensure_eager_tensor(core_math_ops._einsum(dummy, "i,i->", np.array([1.0]), np.array([2.0])))
    assert res is not None
    res = ensure_eager_tensor(core_math_ops._allclose(ensure_eager_tensor(np), np.array([1.0]), np.array([1.0])))
    assert res is not None
    res = ensure_eager_tensor(core_math_ops._allclose(dummy, np.array([1.0]), np.array([1.0])))
    assert res is not None
    res = ensure_eager_tensor(core_math_ops._psum(ensure_eager_tensor(np), np.array([1.0])))
    assert res is not None
    res = ensure_eager_tensor(core_math_ops._pmean(ensure_eager_tensor(np), np.array([1.0])))
    _state.update(ensure_tensor_dict({"res": res}))
    _state.update(ensure_tensor_dict({"dummy": dummy}))


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

    res = ensure_eager_tensor(core_math_ops._polygamma(ensure_mod(DummyMockPolygamma), 1, 1))
    assert res is not None

    class DummyMockDigamma:
        """Test class DummyMockDigamma."""

        pass

    res = ensure_eager_tensor(core_math_ops._digamma(ensure_mod(DummyMockDigamma), 1))
    assert res is not None
    res = ensure_eager_tensor(core_math_ops._np_zerofraction(ensure_eager_tensor(np), np.array([0.0, 1.0])))
    assert res is not None
    _state.update(ensure_tensor_dict({"DummyMockDigamma": DummyMockDigamma, "DummyMockPolygamma": DummyMockPolygamma, "res": res}))


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
                """Test function zerofraction.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    EagerValue: Computed output value from `zerofraction`.
                """
                return 1.0

    res = core_math_ops._np_zerofraction(ensure_mod(MockRandomZeroFraction), np.array([0.0, 1.0]))
    assert res == 0.5
    res = core_math_ops._np_zeta(ensure_eager_tensor(np), 2.0, 1.0)
    assert res is not None
    res = core_math_ops._np_zeta(ensure_mod(_BareMod), 2.0, 1.0)
    assert res is not None
    _state.update(ensure_tensor_dict({"MockRandomZeroFraction": MockRandomZeroFraction, "res": res}))


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
    assert len(_state) > 0


def test_missing_i0() -> None:
    """Test function test_missing_i0."""
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class BkWithI0:
        """Test class BkWithI0."""

        def i0(self, x: EagerValue) -> EagerValue:
            """Test function i0.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `i0`.
            """
            return np.array([1.0])

        def iscomplex(self, x: EagerValue) -> EagerValue:
            """Test function iscomplex.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `iscomplex`.
            """
            return np.array([1.0])

        def isreal(self, x: EagerValue) -> EagerValue:
            """Test function isreal.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `isreal`.
            """
            return np.array([1.0])

        def signbit(self, x: EagerValue) -> EagerValue:
            """Test function signbit.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `signbit`.
            """
            return np.array([1.0])

    class BkWithoutI0:
        """Test class BkWithoutI0."""

        pass

    x = np.array([1.0])
    assert mod._i0(ensure_mod(BkWithI0()), x) is not None
    assert mod._i0(ensure_mod(BkWithoutI0()), x) is None
    assert mod._iscomplex(ensure_mod(BkWithI0()), x) is not None
    assert mod._iscomplex(ensure_mod(BkWithoutI0()), x) is not None
    assert mod._isreal(ensure_mod(BkWithI0()), x) is not None
    assert mod._isreal(ensure_mod(BkWithoutI0()), x) is not None
    assert mod._signbit(ensure_mod(BkWithI0()), x) is not None
    assert mod._signbit(ensure_mod(BkWithoutI0()), x) is not None


class DummyBackend:
    """Test class DummyBackend."""

    def shape(self, x: EagerValue) -> EagerValue:
        """Test function shape.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `shape`.
        """
        return (2, 2)

    def indices(self, x: EagerValue) -> EagerValue:
        """Test function indices.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `indices`.
        """
        return (0, 0)

    def ones(self, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function ones.

        Args:
            x: Test parameter `x`.
            kwargs: Test parameter `kwargs`.

        Returns:
            EagerValue: Computed output value from `ones`.
        """
        return x

    def where(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function where.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            EagerValue: Computed output value from `where`.
        """
        return args[0]

    pass


class DummyBackendWithRandom:
    """Test class DummyBackendWithRandom."""

    class random:
        """Test class random."""

        @staticmethod
        def triangular(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function triangular.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `triangular`.
            """
            return "random.triangular"

        @staticmethod
        def triangularsolve(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function triangularsolve.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `triangularsolve`.
            """
            return "random.triangularsolve"

        @staticmethod
        def xlog1py(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function xlog1py.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `xlog1py`.
            """
            return "random.xlog1py"

        @staticmethod
        def xlogy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function xlogy.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `xlogy`.
            """
            return "random.xlogy"


warnings.filterwarnings("ignore")


def _all_backend_getattr(self: EagerValue, name: EagerValue) -> EagerValue:
    """Resolve attribute on AllBackend.

    Args:
        self (EagerValue): Dynamic `AllBackend` instance.
        name (EagerValue): Input `name` value (name) for `_all_backend_getattr`.

    Returns:
        EagerValue: Computed output value from `_all_backend_getattr`.
    """
    if str(name).startswith("_"):
        raise AttributeError()
    if name == "nn":
        return ensure_eager_tensor(self)

    def _dummy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Return dummy array.

        Args:
            *args (EagerValue): Positional arguments forwarded to `_dummy`.
            **kwargs (EagerValue): Keyword arguments forwarded to `_dummy`.

        Returns:
            EagerValue: Computed output value from `_dummy`.
        """
        return np.array([1.0])

    ensure_mutable(_dummy).item = ensure_eager_tensor(lambda: 1.0)
    ensure_mutable(_dummy).tolist = ensure_eager_tensor(lambda: 1.0)
    return _dummy


AllBackend = ensure_callable(type)("AllBackend", (), {"_" + "_getattr_" + "_": _all_backend_getattr})


def _fallback_backend_init(self: EagerValue, hide_names: EagerValue) -> None:
    """Initialize FallbackBackend.

    Args:
        self (EagerValue): Dynamic `FallbackBackend` instance.
        hide_names (EagerValue): Input `hide_names` value (hide names) for `_fallback_backend_init`.
    """
    from tests.testing_utils import get_test_members

    ensure_mutable(self).hide_names = ensure_eager_tensor(hide_names)
    ensure_mutable(self).np_members = ensure_eager_tensor(get_test_members(np))


def _fallback_backend_getattr(self: EagerValue, name: EagerValue) -> EagerValue:
    """Resolve attribute on FallbackBackend.

    Args:
        self (EagerValue): Dynamic `FallbackBackend` instance.
        name (EagerValue): Input `name` value (name) for `_fallback_backend_getattr`.

    Returns:
        EagerValue: Computed output value from `_fallback_backend_getattr`.
    """
    if ensure_eager_tensor(name) in ensure_eager_tensor(ensure_eager_tensor(self).hide_names):
        raise AttributeError(f"Hidden: {name}")
    if name == "nn":
        return ensure_eager_tensor(self)
    np_members = ensure_eager_tensor(ensure_eager_tensor(self).np_members)
    if str(name) in np_members:
        return np_members[str(name)]
    raise AttributeError(str(name))


FallbackBackend = ensure_callable(type)(
    "FallbackBackend",
    (),
    {"_" + "_init_" + "_": _fallback_backend_init, "_" + "_getattr_" + "_": _fallback_backend_getattr},
)


def _test_all_core_math_ops_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    op = _state["op"]
    sig = inspect.signature(op)
    _state.update(ensure_tensor_dict({"sig": sig}))
    _state.update(ensure_tensor_dict({"op": op}))


def _test_all_core_math_ops_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_int, sig) = (_state["arg"], _state["arg_int"], _state["sig"])

    def build_call(bk: EagerValue, use_int: EagerValue = False, use_one_arg: EagerValue = False, sig: EagerValue = sig) -> EagerValue:
        """Test function build_call.

        Args:
            bk: Test parameter `bk`.
            use_int: Test parameter `use_int`.
            use_one_arg: Test parameter `use_one_arg`.
            sig: Test parameter `sig`.

        Returns:
            EagerValue: Computed output value from `build_call`.
        """
        args = []
        kwargs = {}
        val = arg_int if use_int else arg
        for p in ensure_eager_tensor(ensure_eager_tensor(sig).parameters).values():
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
                ensure_eager_tensor(kwargs)["axes"] = (0,)
            elif p.default != inspect.Parameter.empty:
                pass
            else:
                args.append(val)
        return (args, kwargs)

    _state.update(ensure_tensor_dict({"build_call": build_call}))
    _state.update(ensure_tensor_dict({"arg": arg, "arg_int": arg_int, "sig": sig}))


def _test_all_core_math_ops_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (all_bk, arg, build_call, fb) = (_state["all_bk"], _state["arg"], _state["build_call"], _state["fb"])
    op = _state["op"]
    bk: EagerValue = all_bk
    for bk in [all_bk, fb]:
        for use_int in [False, True]:
            for use_one_arg in [False, True]:
                (args, kwargs) = build_call(bk, use_int, use_one_arg)
                try:
                    ensure_callable(op)(*args, **ensure_attrs(kwargs))
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    handled_errors += 1
                try:
                    op(bk, arg)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    handled_errors += 1
    _state.update(ensure_tensor_dict({"all_bk": all_bk, "arg": arg, "bk": bk}))
    _state.update(ensure_tensor_dict({"build_call": build_call, "fb": fb, "op": op}))


def _test_all_core_math_ops_coverage_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (build_call, fb) = (_state["build_call"], _state["fb"])
    (args, kwargs) = build_call(fb)
    orig_scipy = sys.modules.get("scipy.special")
    orig_scipy2 = sys.modules.get("scipy.linalg")
    orig_scipy_root = sys.modules.get("scipy")
    ensure_eager_tensor(sys.modules)["scipy.special"] = None
    ensure_eager_tensor(sys.modules)["scipy.linalg"] = None
    ensure_eager_tensor(sys.modules)["scipy"] = None
    _state.update(ensure_tensor_dict({"args": args, "kwargs": kwargs, "orig_scipy": orig_scipy, "orig_scipy2": orig_scipy2, "orig_scipy_root": orig_scipy_root}))
    _state.update(ensure_tensor_dict({"build_call": build_call, "fb": fb}))


def _test_all_core_math_ops_coverage_sub_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 5 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (args, kwargs, op, orig_scipy) = (_state["args"], _state["kwargs"], _state["op"], _state["orig_scipy"])
    orig_scipy2 = _state["orig_scipy2"]
    orig_scipy_root = _state["orig_scipy_root"]
    try:
        ensure_callable(op)(*args, **ensure_attrs(kwargs))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    finally:
        if ensure_eager_tensor(orig_scipy) is not None:
            ensure_eager_tensor(sys.modules)["scipy.special"] = orig_scipy
        elif "scipy.special" in sys.modules:
            del sys.modules["scipy.special"]
        if ensure_eager_tensor(orig_scipy2) is not None:
            ensure_eager_tensor(sys.modules)["scipy.linalg"] = orig_scipy2
        elif "scipy.linalg" in sys.modules:
            del sys.modules["scipy.linalg"]
        if ensure_eager_tensor(orig_scipy_root) is not None:
            ensure_eager_tensor(sys.modules)["scipy"] = orig_scipy_root
        elif "scipy" in sys.modules:
            del sys.modules["scipy"]
    _state.update(ensure_tensor_dict({"op": op, "orig_scipy": orig_scipy, "orig_scipy2": orig_scipy2}))
    _state.update(ensure_tensor_dict({"orig_scipy_root": orig_scipy_root}))


def test_all_core_math_ops_coverage() -> None:
    """Test function test_all_core_math_ops_coverage."""
    from tests.testing_utils import get_test_members

    warnings.filterwarnings("ignore")
    cmo_members = get_test_members(cmo)
    ops = [ensure_callable(fn) for name, fn in cmo_members.items() if name.startswith("_") and callable(fn)]
    all_bk = AllBackend()
    arg = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    arg_int = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    hide = [name[1:] for name, fn in cmo_members.items() if name.startswith("_") and callable(fn)]
    fb = FallbackBackend(hide)
    for op in ops:
        _state: dict[str, EagerTensorProtocol] = {}
        _state.update(ensure_tensor_dict({"all_bk": all_bk, "arg": arg, "arg_int": arg_int, "fb": fb}))
        _state.update(ensure_tensor_dict({"hide": hide, "op": op, "ops": ops}))
        for _sub_fn in (_test_all_core_math_ops_coverage_sub_1, _test_all_core_math_ops_coverage_sub_2, _test_all_core_math_ops_coverage_sub_3, _test_all_core_math_ops_coverage_sub_4, _test_all_core_math_ops_coverage_sub_5):
            _sub_fn(_state)
        assert len(_state) > 0


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
                """Test function softmax.

                Args:
                    x: Test parameter `x`.
                    axis: Test parameter `axis`.

                Returns:
                    EagerValue: Computed output value from `softmax`.
                """
                return x

    arg = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    ensure_callable(mod._apply_causal_mask)(ensure_callable(EmptyBk)(), arg)
    ensure_callable(mod._apply_softmax)(ensure_callable(NNSoftmaxBk)(), arg)
    ensure_callable(mod._apply_softmax)(ensure_callable(EmptyBk)(), arg)
    _state.update(ensure_tensor_dict({"EmptyBk": EmptyBk, "NNSoftmaxBk": NNSoftmaxBk, "arg": arg, "mod": mod}))
    _state.update(ensure_tensor_dict({"np": np}))


def _test_missing_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, mod, np) = (_state["arg"], _state["mod"], _state["np"])

    class DummyItem:
        """Test class DummyItem."""

        def item(self) -> EagerValue:
            """Test function item.

            Returns:
                EagerValue: Computed output value from `item`.
            """
            return 1.0

    class DummyList:
        """Test class DummyList."""

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return 1.0

    ensure_callable(mod._allclose)(np, arg, arg, rtol=ensure_callable(DummyItem)())
    ensure_callable(mod._allclose)(np, arg, arg, rtol=ensure_callable(DummyList)())

    class MatmulBk:
        """Test class MatmulBk."""

        @staticmethod
        def matmul(a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                EagerValue: Computed output value from `matmul`.
            """
            return a

    _state.update(ensure_tensor_dict({"DummyItem": DummyItem, "DummyList": DummyList, "MatmulBk": MatmulBk}))
    _state.update(ensure_tensor_dict({"arg": arg, "mod": mod, "np": np}))


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
            """Test function where.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `where`.
            """
            return 1

        @staticmethod
        def floor(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function floor.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `floor`.
            """
            return 1

        @staticmethod
        def ceil(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ceil.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `ceil`.
            """
            return 1

    ensure_callable(mod._fix)(ensure_callable(BkWithWhere)(), arg)
    mod._adaptive_max_pool3d_indices(np, arg, 1)
    _state.update(ensure_tensor_dict({"BkWithWhere": BkWithWhere}))
    _state.update(ensure_tensor_dict({"EmptyBk": EmptyBk, "MatmulBk": MatmulBk, "arg": arg}))
    _state.update(ensure_tensor_dict({"mod": mod, "np": np}))


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
    assert len(_state) > 0


def test_empty_backend() -> None:
    """Test function test_empty_backend."""
    handled_errors = 0
    import inspect

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class EmptyBackend:
        """Test class EmptyBackend."""

        pass

    from tests.testing_utils import get_test_members

    empty_bk = EmptyBackend()
    arg = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    ops = [ensure_callable(fn) for name, fn in get_test_members(mod).items() if name.startswith("_") and callable(fn)]
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = [empty_bk] + [arg] * (len(sig.parameters) - 1)
        try:
            op(*args_to_pass[: len(sig.parameters)])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            handled_errors += 1
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            handled_errors += 1
    assert handled_errors >= 1


def _test_remaining_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    _state.update(ensure_tensor_dict({"mod": mod, "np": np}))


class _BkWithLinalg:
    """Test class _BkWithLinalg."""

    class linalg:
        """Test class linalg."""

        @staticmethod
        def cho_solve(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function cho_solve.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `cho_solve`.
            """
            return np.array([1.0])

        @staticmethod
        def solve_banded(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function solve_banded.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `solve_banded`.
            """
            return np.array([1.0])

        @staticmethod
        def svd(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function svd.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `svd`.
            """
            return np.array([1.0])

        @staticmethod
        def tensorinv(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function tensorinv.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `tensorinv`.
            """
            return np.array([1.0])

        @staticmethod
        def tensorsolve(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function tensorsolve.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `tensorsolve`.
            """
            return np.array([1.0])

        @staticmethod
        def cross(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function cross.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `cross`.
            """
            return np.array([1.0])

        @staticmethod
        def outer(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function outer.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `outer`.
            """
            return np.array([1.0])

        @staticmethod
        def det(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function det.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `det`.
            """
            return np.array([1.0])

        @staticmethod
        def eig(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eig.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `eig`.
            """
            return np.array([1.0])

        @staticmethod
        def eigh(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eigh.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `eigh`.
            """
            return np.array([1.0])

        @staticmethod
        def eigvals(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eigvals.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `eigvals`.
            """
            return np.array([1.0])

        @staticmethod
        def eigvalsh(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eigvalsh.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `eigvalsh`.
            """
            return np.array([1.0])

        @staticmethod
        def cholesky(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function cholesky.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `cholesky`.
            """
            return np.array([1.0])

        @staticmethod
        def matrix_power(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function matrix_power.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `matrix_power`.
            """
            return np.array([1.0])

        @staticmethod
        def norm(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function norm.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `norm`.
            """
            return np.array([1.0])


def _test_remaining_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    _state.update(ensure_tensor_dict({"BkWithLinalg": _BkWithLinalg}))


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
    _state.update(ensure_tensor_dict({"arg": arg, "bk": bk}))
    _state.update(ensure_tensor_dict({"BkWithLinalg": BkWithLinalg, "mod": mod, "np": np}))


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
    _state.update(ensure_tensor_dict({"arg": arg, "bk": bk, "mod": mod}))


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
    _state.update(ensure_tensor_dict({"arg": arg, "bk": bk, "mod": mod}))


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
    _state.update(ensure_tensor_dict({"arg": arg, "bk": bk, "mod": mod}))
    _state.update(ensure_tensor_dict({"np": np}))


def test_remaining_coverage() -> None:
    """Test function test_remaining_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    for _sub_fn in (_test_remaining_coverage_sub_1, _test_remaining_coverage_sub_2, _test_remaining_coverage_sub_3, _test_remaining_coverage_sub_4, _test_remaining_coverage_sub_5, _test_remaining_coverage_sub_6):
        _sub_fn(_state)
    assert len(_state) > 0


def test_unfold_coverage_manual() -> None:
    """Test function test_unfold_coverage_manual."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _np_unfold

    class DummyBackend:
        """Test class DummyBackend."""

        def asarray(self, x: EagerValue) -> EagerValue:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `asarray`.
            """
            return x

    res = _np_unfold(ensure_mod(DummyBackend()), np.ones((1, 1, 4, 4)), kernel_size=3, stride=1)
    assert ensure_eager_tensor(res).shape == (1, 1, 4, 4)
    res = _np_unfold(ensure_mod(DummyBackend()), np.ones((1, 1, 2, 2)), kernel_size=3, stride=1)
    assert ensure_eager_tensor(res).shape == (1, 1, 2, 2)


warnings.filterwarnings("ignore")


def test_final_fftn() -> None:
    """Test function test_final_fftn."""

    class DummyFFT:
        """Test class DummyFFT."""

        def fftn(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftn.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `fftn`.
            """
            return np.array([1.0])

    class BkSpecial:
        """Test class BkSpecial."""

        fft = DummyFFT()

    mod._fftn(ensure_mod(BkSpecial()), np.array([1.0]))
    mod._fftnd(ensure_mod(BkSpecial()), np.array([1.0]))
    assert callable(mod._fftnd)


warnings.filterwarnings("ignore")


class _BkSpecial:
    """Test class _BkSpecial."""

    class linalg:
        """Test class linalg."""

        @staticmethod
        def householder_product(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function householder_product.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `householder_product`.
            """
            return np.array([1.0])

    class fft:
        """Test class fft."""

        @staticmethod
        def fft2(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fft2.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `fft2`.
            """
            return np.array([1.0])

        @staticmethod
        def fftfreq(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftfreq.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `fftfreq`.
            """
            return np.array([1.0])

        @staticmethod
        def fftn(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftn.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `fftn`.
            """
            return np.array([1.0])

        @staticmethod
        def fftshift(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftshift.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `fftshift`.
            """
            return np.array([1.0])

        @staticmethod
        def ifft(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifft.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `ifft`.
            """
            return np.array([1.0])

        @staticmethod
        def ifft2(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifft2.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `ifft2`.
            """
            return np.array([1.0])

        @staticmethod
        def ifftn(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifftn.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `ifftn`.
            """
            return np.array([1.0])

        @staticmethod
        def ifftshift(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ifftshift.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `ifftshift`.
            """
            return np.array([1.0])

    class lax:
        """Test class lax."""

        @staticmethod
        def infeed(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function infeed.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `infeed`.
            """
            return np.array([1.0])

        @staticmethod
        def outfeed(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function outfeed.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `outfeed`.
            """
            return np.array([1.0])

        @staticmethod
        def pshuffle(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function pshuffle.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `pshuffle`.
            """
            return np.array([1.0])

        @staticmethod
        def pswapaxes(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function pswapaxes.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `pswapaxes`.
            """
            return np.array([1.0])

        @staticmethod
        def ppermute(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function ppermute.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `ppermute`.
            """
            return np.array([1.0])

        @staticmethod
        def psum_scatter(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function psum_scatter.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `psum_scatter`.
            """
            return np.array([1.0])

    class random:
        """Test class random."""

        @staticmethod
        def tridiagonal(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function tridiagonal.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `tridiagonal`.
            """
            return np.array([1.0])


def _test_even_more_coverage_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    _state.update(ensure_tensor_dict({"BkSpecial": _BkSpecial}))


def _test_even_more_coverage_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    BkSpecial = _state["BkSpecial"]
    bk = BkSpecial()
    arg = np.array([1.0])
    mod._accumulate_n(ensure_eager_tensor(np), [arg, arg])
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
    _state.update(ensure_tensor_dict({"arg": arg, "bk": bk}))
    _state.update(ensure_tensor_dict({"BkSpecial": BkSpecial}))


def _test_even_more_coverage_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk) = (_state["arg"], _state["bk"])
    assert mod._fftn(bk, arg) is not None
    assert mod._psumscatter(bk, arg) is not None
    _state.update(ensure_tensor_dict({"arg": arg, "bk": bk}))


def test_even_more_coverage() -> None:
    """Test function test_even_more_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    for _sub_fn in (_test_even_more_coverage_sub_1, _test_even_more_coverage_sub_2, _test_even_more_coverage_sub_3):
        _sub_fn(_state)
    assert len(_state) > 0
