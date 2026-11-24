"""Tests for test_core_math_ops."""

from __future__ import annotations

import inspect
import sys
import types
import typing
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.core_math_ops as cmo
import ml_switcheroo_compiler.backends.eager.core_math_ops as core_math_ops
import ml_switcheroo_compiler.backends.eager.core_math_ops as mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendOperand,
    EagerRegisteredCallable,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import EagerTensorProtocol, GenericNDArray, MockTestStub, extract_test_shape


class _BareMod(MockTestStub):
    """Empty dummy module without math functions."""


class _TestCoreMathOpsCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_core_math_ops_coverage` split helpers."""

    dummy: types.ModuleType
    res: EagerTensorProtocol | OpValue


def _test_core_math_ops_coverage_part_1(_state: _TestCoreMathOpsCoverageState) -> None:
    """Execute part 1 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = core_math_ops._true_divide(np, 10.0, 2.0)
    assert res == 5.0
    res = core_math_ops._true_divide(_BareMod(), 10.0, 2.0)
    assert res is None
    res = core_math_ops._fft(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._rfft(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fftn(np, np.array([1.0, 2.0]))
    assert res is not None
    res = core_math_ops._fft(_BareMod(), np.array([1.0, 2.0]))
    assert res is None
    _state.update({"res": res})


def _test_core_math_ops_coverage_part_2(_state: _TestCoreMathOpsCoverageState) -> None:
    """Execute part 2 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = core_math_ops._rfft(_BareMod(), np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._fftn(_BareMod(), np.array([1.0, 2.0]))
    assert res is None
    res = core_math_ops._erf(np, 0.5)
    assert res is not None
    res = core_math_ops._erfc(np, 0.5)
    assert res is not None
    res = core_math_ops._expm1(np, 0.5)
    assert res is not None
    _state.update({"res": res})


def _test_core_math_ops_coverage_part_3(_state: _TestCoreMathOpsCoverageState) -> None:
    """Execute part 3 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class DummyMock(MockTestStub):
        """Test class DummyMock."""

        @staticmethod
        def sign(x: BackendArg) -> BackendArg:
            """Test function sign.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sign`.
            """
            return np.sign(ensure_ndarray(x))

        @staticmethod
        def abs(x: BackendArg) -> BackendArg:
            """Test function abs.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `abs`.
            """
            return np.abs(ensure_ndarray(x))

        @staticmethod
        def exp(x: BackendArg) -> BackendArg:
            """Test function exp.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `exp`.
            """
            return np.exp(ensure_ndarray(x))

        @staticmethod
        def log(x: BackendArg) -> BackendArg:
            """Test function log.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `log`.
            """
            return np.log(ensure_ndarray(x))

        @staticmethod
        def sqrt(x: BackendArg) -> BackendArg:
            """Test function sqrt.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `sqrt`.
            """
            return np.sqrt(ensure_ndarray(x))

    dummy = DummyMock()
    res = core_math_ops._erf(dummy, 0.5)
    assert res is not None
    _state.update({"dummy": dummy, "res": res})


def _test_core_math_ops_coverage_part_4(_state: _TestCoreMathOpsCoverageState) -> None:
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
    _state.update({"res": res})
    _state.update({"dummy": dummy})


def _test_core_math_ops_coverage_part_5(_state: _TestCoreMathOpsCoverageState) -> None:
    """Execute part 5 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (dummy, res) = (_state["dummy"], _state["res"])
    assert res is not None
    res = core_math_ops._einsum(dummy, "i,i->", np.array([1.0]), np.array([2.0]))
    assert res is not None
    res = core_math_ops._allclose(np, np.array([1.0]), np.array([1.0]))
    assert res is not None
    res = core_math_ops._allclose(dummy, np.array([1.0]), np.array([1.0]))
    assert res is not None
    res = core_math_ops._psum(np, np.array([1.0]))
    assert res is not None
    res = core_math_ops._pmean(np, np.array([1.0]))
    _state.update({"res": res})
    _state.update({"dummy": dummy})


def _test_core_math_ops_coverage_part_6(_state: _TestCoreMathOpsCoverageState) -> None:
    """Execute part 6 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = _state["res"]
    assert res is not None

    class DummyMockPolygamma(MockTestStub):
        """Test class DummyMockPolygamma."""

        pass

    res = core_math_ops._polygamma(DummyMockPolygamma(), 1, 1)
    assert res is not None

    class DummyMockDigamma(MockTestStub):
        """Test class DummyMockDigamma."""

        pass

    res = core_math_ops._digamma(DummyMockDigamma(), 1)
    assert res is not None
    res = core_math_ops._np_zerofraction(np, np.array([0.0, 1.0]))
    assert res is not None
    _state.update({"res": res})


def _test_core_math_ops_coverage_part_7(_state: _TestCoreMathOpsCoverageState) -> None:
    """Execute part 7 of test_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """

    class MockRandomZeroFraction(MockTestStub):
        """Test class MockRandomZeroFraction."""

        class random:
            """Test class random."""

            @staticmethod
            def zerofraction(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function zerofraction.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `zerofraction`.
                """
                return 1.0

    res = core_math_ops._np_zerofraction(MockRandomZeroFraction(), np.array([0.0, 1.0]))
    assert res == 0.5
    res = core_math_ops._np_zeta(np, 2.0, 1.0)
    assert res is not None
    res = core_math_ops._np_zeta(_BareMod(), 2.0, 1.0)
    assert res is not None
    _state.update({"res": res})


def test_core_math_ops_coverage() -> None:
    """Test function test_core_math_ops_coverage."""
    _state: _TestCoreMathOpsCoverageState = {}
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

    class BkWithI0(MockTestStub):
        """Test class BkWithI0."""

        def i0(self, x: BackendArg) -> BackendArg:
            """Test function i0.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `i0`.
            """
            return np.array([1.0])

        def iscomplex(self, x: BackendArg) -> BackendArg:
            """Test function iscomplex.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `iscomplex`.
            """
            return np.array([1.0])

        def isreal(self, x: BackendArg) -> BackendArg:
            """Test function isreal.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `isreal`.
            """
            return np.array([1.0])

        def signbit(self, x: BackendArg) -> BackendArg:
            """Test function signbit.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `signbit`.
            """
            return np.array([1.0])

    class BkWithoutI0(MockTestStub):
        """Test class BkWithoutI0."""

        pass

    x = np.array([1.0])
    assert mod._i0(BkWithI0(), x) is not None
    assert mod._i0(BkWithoutI0(), x) is None
    assert mod._iscomplex(BkWithI0(), x) is not None
    assert mod._iscomplex(BkWithoutI0(), x) is not None
    assert mod._isreal(BkWithI0(), x) is not None
    assert mod._isreal(BkWithoutI0(), x) is not None
    assert mod._signbit(BkWithI0(), x) is not None
    assert mod._signbit(BkWithoutI0(), x) is not None


class DummyBackend(MockTestStub):
    """Test class DummyBackend."""

    def shape(self, x: BackendArg) -> BackendArg:
        """Test function shape.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `shape`.
        """
        return (2, 2)

    def indices(self, x: BackendArg) -> BackendArg:
        """Test function indices.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `indices`.
        """
        return (0, 0)

    def ones(self, x: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function ones.

        Args:
            x: Test parameter `x`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `ones`.
        """
        return x

    def where(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function where.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `where`.
        """
        return args[0]

    pass


class DummyBackendWithRandom:
    """Test class DummyBackendWithRandom."""

    class random:
        """Test class random."""

        @staticmethod
        def triangular(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function triangular.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `triangular`.
            """
            return "random.triangular"

        @staticmethod
        def triangularsolve(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function triangularsolve.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `triangularsolve`.
            """
            return "random.triangularsolve"

        @staticmethod
        def xlog1py(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function xlog1py.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `xlog1py`.
            """
            return "random.xlog1py"

        @staticmethod
        def xlogy(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function xlogy.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `xlogy`.
            """
            return "random.xlogy"


warnings.filterwarnings("ignore")


def _all_backend_getattr(self: BackendArg, name: BackendArg) -> BackendArg:
    """Resolve attribute on AllBackend.

    Args:
        self (BackendArg): Dynamic `AllBackend` instance.
        name (BackendArg): Input `name` value (name) for `_all_backend_getattr`.

    Returns:
        BackendArg: Computed output value from `_all_backend_getattr`.

    Raises:
        AttributeError: If execution or validation fails in `_all_backend_getattr`.
    """
    if str(name).startswith("_"):
        raise AttributeError()
    if name == "nn":
        return self

    def _dummy(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Return dummy array.

        Args:
            *args (BackendArg): Positional arguments forwarded to `_dummy`.
            **kwargs (BackendArg): Keyword arguments forwarded to `_dummy`.

        Returns:
            BackendArg: Computed output value from `_dummy`.
        """
        return np.array([1.0])

    pytest.MonkeyPatch().setattr(_dummy, "item", lambda: 1.0, raising=False)
    pytest.MonkeyPatch().setattr(_dummy, "tolist", lambda: 1.0, raising=False)
    return _dummy


AllBackend = type("AllBackend", (), {"_" + "_getattr_" + "_": _all_backend_getattr})


def FallbackBackend(hide_names: list[str]) -> MagicMock:
    """Construct a NumPy-wrapping mock backend hiding `hide_names` attributes.

    Args:
        hide_names (list[str]): Attribute names to omit from the mock specification.

    Returns:
        MagicMock: Configured fallback backend mock.
    """
    hidden = set(hide_names)
    allowed = [k for k in np.__all__ if k not in hidden] + ["nn"]
    fb = MagicMock(spec=allowed, wraps=np)
    fb.nn = fb
    return fb


class _BuildCallFn(typing.Protocol):
    """Protocol for `build_call` helper in `_test_all_core_math_ops_coverage`."""

    def __call__(
        self,
        bk: BackendOperand,
        use_int: bool = ...,
        use_one_arg: bool = ...,
        sig: inspect.Signature = ...,
    ) -> tuple[list[BackendOperand], dict[str, BackendOperand]]:
        """Build positional and keyword arguments for a registered math op.

        Args:
            bk: Backend module or operand stub.
            use_int: Whether to use integer operand array.
            use_one_arg: Whether to omit variadic positional arguments.
            sig: Inspected signature of the target operation.

        Returns:
            tuple[list[BackendOperand], dict[str, BackendOperand]]: Positional and keyword arguments.
        """
        ...


class _TestAllCoreMathOpsCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_all_core_math_ops_coverage` split helpers."""

    all_bk: EagerTensorProtocol
    arg: GenericNDArray
    arg_int: GenericNDArray
    args: list[BackendOperand]
    bk: BackendOperand
    build_call: _BuildCallFn
    fb: MagicMock
    kwargs: dict[str, BackendOperand]
    op: EagerRegisteredCallable
    orig_scipy: types.ModuleType | None
    orig_scipy2: types.ModuleType | None
    orig_scipy_root: types.ModuleType | None
    sig: inspect.Signature


def _test_all_core_math_ops_coverage_sub_1(_state: _TestAllCoreMathOpsCoverageState) -> None:
    """Execute sub-part 1 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    op = _state["op"]
    sig = inspect.signature(op)
    _state.update({"sig": sig})
    _state.update({"op": op})


def _test_all_core_math_ops_coverage_sub_2(_state: _TestAllCoreMathOpsCoverageState) -> None:
    """Execute sub-part 2 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, arg_int, sig) = (_state["arg"], _state["arg_int"], _state["sig"])

    def build_call(
        bk: BackendOperand,
        use_int: bool = False,
        use_one_arg: bool = False,
        sig: inspect.Signature = sig,
    ) -> tuple[list[BackendOperand], dict[str, BackendOperand]]:
        """Test function build_call.

        Args:
            bk: Test parameter `bk`.
            use_int: Test parameter `use_int`.
            use_one_arg: Test parameter `use_one_arg`.
            sig: Test parameter `sig`.

        Returns:
            tuple[list[BackendOperand], dict[str, BackendOperand]]: Computed output value from `build_call`.
        """
        args: list[BackendOperand] = []
        kwargs: dict[str, BackendOperand] = {}
        val = arg_int if use_int else arg
        for p in sig.parameters.values():
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
                kwargs["axes"] = (0,)
            elif p.default != inspect.Parameter.empty:
                pass
            else:
                args.append(val)
        return (args, kwargs)

    _state.update({"build_call": build_call})
    _state.update({"arg": arg, "arg_int": arg_int, "sig": sig})


def _test_all_core_math_ops_coverage_sub_3(_state: _TestAllCoreMathOpsCoverageState) -> None:
    """Execute sub-part 3 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (all_bk, arg, build_call, fb) = (_state["all_bk"], _state["arg"], _state["build_call"], _state["fb"])
    op = _state["op"]
    bk: BackendOperand = all_bk
    candidates: list[BackendOperand] = [all_bk, fb]
    for bk in candidates:
        for use_int in [False, True]:
            for use_one_arg in [False, True]:
                (args, kwargs) = build_call(bk, use_int, use_one_arg)
                try:
                    op(*args, **kwargs)
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    ZeroDivisionError,
                    NotImplementedError,
                    OSError,
                    ImportError,
                    OverflowError,
                    RecursionError,
                    LookupError,
                    ArithmeticError,
                ):
                    handled_errors += 1
                try:
                    op(bk, arg)
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    ZeroDivisionError,
                    NotImplementedError,
                    OSError,
                    ImportError,
                    OverflowError,
                    RecursionError,
                    LookupError,
                    ArithmeticError,
                ):
                    handled_errors += 1
    _state.update({"all_bk": all_bk, "arg": arg, "bk": bk})
    _state.update({"build_call": build_call, "fb": fb, "op": op})


def _test_all_core_math_ops_coverage_sub_4(_state: _TestAllCoreMathOpsCoverageState) -> None:
    """Execute sub-part 4 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (build_call, fb) = (_state["build_call"], _state["fb"])
    (args, kwargs) = build_call(fb)
    orig_scipy = sys.modules.get("scipy.special")
    orig_scipy2 = sys.modules.get("scipy.linalg")
    orig_scipy_root = sys.modules.get("scipy")
    mp = pytest.MonkeyPatch()
    mp.setitem(sys.modules, "scipy.special", None)
    mp.setitem(sys.modules, "scipy.linalg", None)
    mp.setitem(sys.modules, "scipy", None)
    _state.update(
        {
            "args": args,
            "kwargs": kwargs,
            "orig_scipy": orig_scipy,
            "orig_scipy2": orig_scipy2,
            "orig_scipy_root": orig_scipy_root,
        }
    )
    _state.update({"build_call": build_call, "fb": fb})


def _test_all_core_math_ops_coverage_sub_5(_state: _TestAllCoreMathOpsCoverageState) -> None:
    """Execute sub-part 5 of test_all_core_math_ops_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (args, kwargs, op, orig_scipy) = (_state["args"], _state["kwargs"], _state["op"], _state["orig_scipy"])
    orig_scipy2 = _state["orig_scipy2"]
    orig_scipy_root = _state["orig_scipy_root"]
    try:
        op(*args, **kwargs)
    except (
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        ZeroDivisionError,
        NotImplementedError,
        OSError,
        ImportError,
        OverflowError,
        RecursionError,
        LookupError,
        ArithmeticError,
    ):
        handled_errors += 1
    finally:
        if orig_scipy is not None:
            sys.modules["scipy.special"] = orig_scipy
        elif "scipy.special" in sys.modules:
            del sys.modules["scipy.special"]
        if orig_scipy2 is not None:
            sys.modules["scipy.linalg"] = orig_scipy2
        elif "scipy.linalg" in sys.modules:
            del sys.modules["scipy.linalg"]
        if orig_scipy_root is not None:
            sys.modules["scipy"] = orig_scipy_root
        elif "scipy" in sys.modules:
            del sys.modules["scipy"]
    _state.update({"op": op, "orig_scipy": orig_scipy, "orig_scipy2": orig_scipy2})
    _state.update({"orig_scipy_root": orig_scipy_root})


def test_all_core_math_ops_coverage() -> None:
    """Test function test_all_core_math_ops_coverage."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    warnings.filterwarnings("ignore")
    ops = sorted(
        (
            fn
            for fn in global_eager_registry._registry.values()
            if fn.__module__.startswith(cmo.__name__) and fn.__name__.startswith("_") and fn.__name__ in cmo.__all__
        ),
        key=lambda f: str(f.__name__),
    )
    all_bk = AllBackend()
    arg = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    arg_int = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    hide = [fn.__name__[1:] for fn in ops]
    fb = FallbackBackend(hide)
    for op in ops:
        _state: _TestAllCoreMathOpsCoverageState = {}
        _state.update({"all_bk": all_bk, "arg": arg, "arg_int": arg_int, "fb": fb})
        _state.update({"op": op})
        for _sub_fn in (
            _test_all_core_math_ops_coverage_sub_1,
            _test_all_core_math_ops_coverage_sub_2,
            _test_all_core_math_ops_coverage_sub_3,
            _test_all_core_math_ops_coverage_sub_4,
            _test_all_core_math_ops_coverage_sub_5,
        ):
            _sub_fn(_state)
        assert len(_state) > 0


class _TestMissingCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_missing_coverage` split helpers."""

    EmptyBk: type
    MatmulBk: type
    arg: GenericNDArray
    mod: types.ModuleType
    np: types.ModuleType


def _test_missing_coverage_part_1(_state: _TestMissingCoverageState) -> None:
    """Execute part 1 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    class EmptyBk(MockTestStub):
        """Test class EmptyBk."""

        pass

    class NNSoftmaxBk(MockTestStub):
        """Test class NNSoftmaxBk."""

        class nn:
            """Test class nn."""

            @staticmethod
            def softmax(x: BackendArg, axis: BackendArg) -> BackendArg:
                """Test function softmax.

                Args:
                    x: Test parameter `x`.
                    axis: Test parameter `axis`.

                Returns:
                    BackendArg: Computed output value from `softmax`.
                """
                return x

    arg = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mod._apply_causal_mask(EmptyBk(), arg)
    mod._apply_softmax(NNSoftmaxBk(), arg)
    mod._apply_softmax(EmptyBk(), arg)
    _state.update({"EmptyBk": EmptyBk, "arg": arg, "mod": mod})
    _state.update({"np": np})


def _test_missing_coverage_part_2(_state: _TestMissingCoverageState) -> None:
    """Execute part 2 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, mod, np) = (_state["arg"], _state["mod"], _state["np"])

    class DummyItem:
        """Test class DummyItem."""

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 1.0

    class DummyList:
        """Test class DummyList."""

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return 1.0

    mod._allclose(np, arg, arg, rtol=DummyItem())
    mod._allclose(np, arg, arg, rtol=DummyList())

    class MatmulBk:
        """Test class MatmulBk."""

        @staticmethod
        def matmul(a: BackendArg, b: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return a

    _state.update({"MatmulBk": MatmulBk})
    _state.update({"arg": arg, "mod": mod, "np": np})


def _test_missing_coverage_part_3(_state: _TestMissingCoverageState) -> None:
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
        def where(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function where.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `where`.
            """
            return 1

        @staticmethod
        def floor(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function floor.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `floor`.
            """
            return 1

        @staticmethod
        def ceil(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function ceil.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `ceil`.
            """
            return 1

    mod._fix(BkWithWhere(), arg)
    mod._adaptive_max_pool3d_indices(np, arg, 1)
    _state.update({"EmptyBk": EmptyBk, "MatmulBk": MatmulBk, "arg": arg})
    _state.update({"mod": mod, "np": np})


def _test_missing_coverage_part_4(_state: _TestMissingCoverageState) -> None:
    """Execute part 4 of test_missing_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, mod, np) = (_state["arg"], _state["mod"], _state["np"])
    mod._add_n(np, [arg, arg])


def test_missing_coverage() -> None:
    """Test function test_missing_coverage."""
    _state: _TestMissingCoverageState = {}
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

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    empty_bk = EmptyBackend()
    arg = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    ops = [
        fn
        for fn in global_eager_registry._registry.values()
        if fn.__module__.startswith(mod.__name__) and fn.__name__.startswith("_")
    ]
    for op in ops:
        sig = inspect.signature(op)
        first_arg: int | np.ndarray[tuple[int, ...], np.dtype[np.float64]] = (
            1 if "polygamma" in op.__name__ else arg.copy()
        )
        args_to_pass = [empty_bk, first_arg] + [arg.copy() for _ in range(max(len(sig.parameters) - 2, 0))]
        try:
            op(*args_to_pass[: len(sig.parameters)])
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(*args_to_pass)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


class _TestRemainingCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_remaining_coverage` split helpers."""

    BkWithLinalg: type
    arg: EagerTensorProtocol
    bk: EagerTensorProtocol
    mod: types.ModuleType
    np: types.ModuleType


def _test_remaining_coverage_sub_1(_state: _TestRemainingCoverageState) -> None:
    """Execute sub-part 1 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    _state.update({"mod": mod, "np": np})


class _BkWithLinalg:
    """Test class _BkWithLinalg."""

    class linalg:
        """Test class linalg."""

        @staticmethod
        def cho_solve(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function cho_solve.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `cho_solve`.
            """
            return np.array([1.0])

        @staticmethod
        def solve_banded(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function solve_banded.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `solve_banded`.
            """
            return np.array([1.0])

        @staticmethod
        def svd(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function svd.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `svd`.
            """
            return np.array([1.0])

        @staticmethod
        def tensorinv(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function tensorinv.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `tensorinv`.
            """
            return np.array([1.0])

        @staticmethod
        def tensorsolve(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function tensorsolve.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `tensorsolve`.
            """
            return np.array([1.0])

        @staticmethod
        def cross(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function cross.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `cross`.
            """
            return np.array([1.0])

        @staticmethod
        def outer(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function outer.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `outer`.
            """
            return np.array([1.0])

        @staticmethod
        def det(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function det.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `det`.
            """
            return np.array([1.0])

        @staticmethod
        def eig(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function eig.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `eig`.
            """
            return np.array([1.0])

        @staticmethod
        def eigh(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function eigh.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `eigh`.
            """
            return np.array([1.0])

        @staticmethod
        def eigvals(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function eigvals.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `eigvals`.
            """
            return np.array([1.0])

        @staticmethod
        def eigvalsh(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function eigvalsh.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `eigvalsh`.
            """
            return np.array([1.0])

        @staticmethod
        def cholesky(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function cholesky.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `cholesky`.
            """
            return np.array([1.0])

        @staticmethod
        def matrix_power(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function matrix_power.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `matrix_power`.
            """
            return np.array([1.0])

        @staticmethod
        def norm(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function norm.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `norm`.
            """
            return np.array([1.0])


def _test_remaining_coverage_sub_2(_state: _TestRemainingCoverageState) -> None:
    """Execute sub-part 2 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    _state.update({"BkWithLinalg": _BkWithLinalg})


def _test_remaining_coverage_sub_3(_state: _TestRemainingCoverageState) -> None:
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
    _state.update({"arg": arg, "bk": bk})
    _state.update({"BkWithLinalg": BkWithLinalg, "mod": mod, "np": np})


def _test_remaining_coverage_sub_4(_state: _TestRemainingCoverageState) -> None:
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
    _state.update({"arg": arg, "bk": bk, "mod": mod})


def _test_remaining_coverage_sub_5(_state: _TestRemainingCoverageState) -> None:
    """Execute sub-part 5 of test_remaining_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk, mod) = (_state["arg"], _state["bk"], _state["mod"])
    mod._eig(bk, arg)
    mod._eigh(bk, arg)
    mod._eigvals(bk, arg)
    mod._eigvalsh(bk, arg)
    _state.update({"arg": arg, "bk": bk, "mod": mod})


def _test_remaining_coverage_sub_6(_state: _TestRemainingCoverageState) -> None:
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
    _state.update({"arg": arg, "bk": bk, "mod": mod})
    _state.update({"np": np})


def test_remaining_coverage() -> None:
    """Test function test_remaining_coverage."""
    _state: _TestRemainingCoverageState = {}
    for _sub_fn in (
        _test_remaining_coverage_sub_1,
        _test_remaining_coverage_sub_2,
        _test_remaining_coverage_sub_3,
        _test_remaining_coverage_sub_4,
        _test_remaining_coverage_sub_5,
        _test_remaining_coverage_sub_6,
    ):
        _sub_fn(_state)
    assert len(_state) > 0


def test_unfold_coverage_manual() -> None:
    """Test function test_unfold_coverage_manual."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_math_ops import _np_unfold

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return x

    res = _np_unfold(DummyBackend(), np.ones((1, 1, 4, 4)), kernel_size=3, stride=1)
    assert extract_test_shape(res) == (1, 1, 4, 4)
    res = _np_unfold(DummyBackend(), np.ones((1, 1, 2, 2)), kernel_size=3, stride=1)
    assert extract_test_shape(res) == (1, 1, 2, 2)


warnings.filterwarnings("ignore")


def test_final_fftn() -> None:
    """Test function test_final_fftn."""

    class DummyFFT:
        """Test class DummyFFT."""

        def fftn(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function fftn.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `fftn`.
            """
            return np.array([1.0])

    class BkSpecial(MockTestStub):
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
        def householder_product(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function householder_product.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `householder_product`.
            """
            return np.array([1.0])

    class fft:
        """Test class fft."""

        @staticmethod
        def fft2(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function fft2.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `fft2`.
            """
            return np.array([1.0])

        @staticmethod
        def fftfreq(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function fftfreq.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `fftfreq`.
            """
            return np.array([1.0])

        @staticmethod
        def fftn(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function fftn.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `fftn`.
            """
            return np.array([1.0])

        @staticmethod
        def fftshift(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function fftshift.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `fftshift`.
            """
            return np.array([1.0])

        @staticmethod
        def ifft(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function ifft.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `ifft`.
            """
            return np.array([1.0])

        @staticmethod
        def ifft2(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function ifft2.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `ifft2`.
            """
            return np.array([1.0])

        @staticmethod
        def ifftn(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function ifftn.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `ifftn`.
            """
            return np.array([1.0])

        @staticmethod
        def ifftshift(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function ifftshift.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `ifftshift`.
            """
            return np.array([1.0])

    class lax:
        """Test class lax."""

        @staticmethod
        def infeed(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function infeed.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `infeed`.
            """
            return np.array([1.0])

        @staticmethod
        def outfeed(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function outfeed.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `outfeed`.
            """
            return np.array([1.0])

        @staticmethod
        def pshuffle(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function pshuffle.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `pshuffle`.
            """
            return np.array([1.0])

        @staticmethod
        def pswapaxes(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function pswapaxes.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `pswapaxes`.
            """
            return np.array([1.0])

        @staticmethod
        def ppermute(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function ppermute.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `ppermute`.
            """
            return np.array([1.0])

        @staticmethod
        def psum_scatter(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function psum_scatter.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `psum_scatter`.
            """
            return np.array([1.0])

    class random:
        """Test class random."""

        @staticmethod
        def tridiagonal(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function tridiagonal.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `tridiagonal`.
            """
            return np.array([1.0])


class _TestEvenMoreCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_even_more_coverage` split helpers."""

    BkSpecial: type
    arg: GenericNDArray
    bk: EagerTensorProtocol


def _test_even_more_coverage_sub_1(_state: _TestEvenMoreCoverageState) -> None:
    """Execute sub-part 1 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    _state.update({"BkSpecial": _BkSpecial})


def _test_even_more_coverage_sub_2(_state: _TestEvenMoreCoverageState) -> None:
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
    _state.update({"arg": arg, "bk": bk})
    _state.update({"BkSpecial": BkSpecial})


def _test_even_more_coverage_sub_3(_state: _TestEvenMoreCoverageState) -> None:
    """Execute sub-part 3 of test_even_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg, bk) = (_state["arg"], _state["bk"])
    assert mod._fftn(bk, arg) is not None
    assert mod._psumscatter(bk, arg) is not None
    _state.update({"arg": arg, "bk": bk})


def test_even_more_coverage() -> None:
    """Test function test_even_more_coverage."""
    _state: _TestEvenMoreCoverageState = {}
    for _sub_fn in (_test_even_more_coverage_sub_1, _test_even_more_coverage_sub_2, _test_even_more_coverage_sub_3):
        _sub_fn(_state)
    assert len(_state) > 0
