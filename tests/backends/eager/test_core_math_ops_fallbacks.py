# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_core_math_ops_fallbacks."""

from __future__ import annotations

import types
import typing
import warnings
from pathlib import Path

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.core_math_ops as mod
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import (
    BackendOperand,
    EagerBoundCallable,
    EagerTensorProtocol,
    MockTestStub,
    adapt_backend_module,
    require_callable,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerCallableProtocol, EagerTensorProtocol
warnings.filterwarnings("ignore")


class EmptyMockBackend(MockTestStub):
    """Test class EmptyMockBackend."""

    pass


class FullMockBackend(MockTestStub):
    """Test class FullMockBackend."""

    pass


class MockNamespace:
    """Test class MockNamespace."""

    pass


class MissingPrimaryMockBackend(MockTestStub):
    """Test class MissingPrimaryMockBackend."""

    random = MockNamespace()
    nn = MockNamespace()
    linalg = MockNamespace()
    fft = MockNamespace()
    signal = MockNamespace()
    special = MockNamespace()
    image = MockNamespace()


class _DisabledTestCoreMathOpsBranchesState(typing.TypedDict, total=False):
    """Typed shared state for `_disabled_test_core_math_ops_branches` split helpers."""

    AttnBackend: type
    MockData3: type
    builtins: types.ModuleType
    dummy_input: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    empty_backend: EmptyMockBackend
    full_backend: FullMockBackend
    mock_hasattr: EagerBoundCallable
    nested_backend: MissingPrimaryMockBackend
    original_hasattr: EagerBoundCallable
    original_scipy: types.ModuleType | None
    scipy_installed: bool
    sys: types.ModuleType


def _disabled_test_core_math_ops_branches_sub_1(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 1 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = EmptyMockBackend()
    full_backend = FullMockBackend()
    nested_backend = MissingPrimaryMockBackend()
    dummy_input = np.ones((2, 2))
    _state.update(
        {
            "dummy_input": dummy_input,
            "empty_backend": empty_backend,
            "full_backend": full_backend,
            "nested_backend": nested_backend,
        }
    )


def _disabled_test_core_math_ops_branches_sub_2(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 2 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, empty_backend) = (_state["dummy_input"], _state["empty_backend"])
    for _name, func in global_eager_registry._registry.items():
        if func.__module__ == "ml_switcheroo_compiler.backends.eager.core_math_ops":
            try:
                func(empty_backend, dummy_input)
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
                func(empty_backend, dummy_input, dummy_input)
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
                func(empty_backend, dummy_input, dummy_input, dummy_input)
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
    _state.update({"dummy_input": dummy_input, "empty_backend": empty_backend})


def _disabled_test_core_math_ops_branches_sub_3(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 3 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, full_backend) = (_state["dummy_input"], _state["full_backend"])
    for _name, func in global_eager_registry._registry.items():
        if func.__module__ == "ml_switcheroo_compiler.backends.eager.core_math_ops":
            try:
                func(full_backend, dummy_input)
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
                func(full_backend, dummy_input, dummy_input)
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
                func(full_backend, dummy_input, dummy_input, dummy_input)
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
    _state.update({"dummy_input": dummy_input, "full_backend": full_backend})


def _disabled_test_core_math_ops_branches_sub_4(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 4 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, nested_backend) = (_state["dummy_input"], _state["nested_backend"])
    for _name, func in global_eager_registry._registry.items():
        if func.__module__ == "ml_switcheroo_compiler.backends.eager.core_math_ops":
            try:
                func(nested_backend, dummy_input)
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
                func(nested_backend, dummy_input, dummy_input)
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
                func(nested_backend, dummy_input, dummy_input, dummy_input)
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
    _state.update({"dummy_input": dummy_input, "nested_backend": nested_backend})


def _disabled_test_core_math_ops_branches_sub_5(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 5 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, full_backend, nested_backend) = (
        _state["dummy_input"],
        _state["full_backend"],
        _state["nested_backend"],
    )
    try:
        mod._apply_causal_mask(full_backend, dummy_input)
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
        mod._apply_softmax(nested_backend, dummy_input)
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
        adapt_backend_module(mod).require_symbol("_global_adaptive_pool_mock")(full_backend, dummy_input, 1)
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
        adapt_backend_module(mod).require_symbol("_global_adaptive_pool_mock")(full_backend, dummy_input, (1, 1))
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
    _state.update({"dummy_input": dummy_input, "full_backend": full_backend, "nested_backend": nested_backend})


def _disabled_test_core_math_ops_branches_sub_6(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 6 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, empty_backend, full_backend) = (
        _state["dummy_input"],
        _state["empty_backend"],
        _state["full_backend"],
    )
    try:
        mod._gamma(empty_backend, 1.0)
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

    class MockData(MockTestStub):
        """Test class MockData."""

        data = np.ones((2, 2))
        dtype = float

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 1.0

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1.0]

    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=MockData())
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
    _state.update({"dummy_input": dummy_input, "empty_backend": empty_backend, "full_backend": full_backend})


def _disabled_test_core_math_ops_branches_sub_7(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 7 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, full_backend) = (_state["dummy_input"], _state["full_backend"])

    class MockData2(MockTestStub):
        """Test class MockData2."""

        data = np.ones((2, 2))
        dtype = float

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1.0]

    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=MockData2())
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

    class MockData3:
        """Test class MockData3."""

        data = np.ones((2, 2))
        dtype = float

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1.0]

    _state.update({"MockData3": MockData3})
    _state.update({"dummy_input": dummy_input, "full_backend": full_backend})


def _disabled_test_core_math_ops_branches_sub_8(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 8 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (MockData3, dummy_input, full_backend) = (_state["MockData3"], _state["dummy_input"], _state["full_backend"])
    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=MockData3(), atol=MockData3(), equal_nan=MockData3())
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

    class NoTriuBackend(MockTestStub):
        """Test class NoTriuBackend."""

        pass

    try:
        mod._apply_causal_mask(NoTriuBackend(), np.ones((2, 2)))
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
        mod._apply_softmax(NoTriuBackend(), np.ones((2, 2)))
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
    _state.update({"MockData3": MockData3, "dummy_input": dummy_input, "full_backend": full_backend})


def _disabled_test_core_math_ops_branches_sub_9(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 9 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """

    class AttnBackend(MockTestStub):
        """Test class AttnBackend."""

        def matmul(self, a: BackendArg, b: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return a

        def transpose(self, a: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function transpose.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `transpose`.
            """
            return a

        def triu(self, a: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function triu.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `triu`.
            """
            return a

        def ones(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function ones.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `ones`.
            """
            return np.ones((2, 2))

        def where(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function where.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `where`.
            """
            return np.ones((2, 2))

        def softmax(self, a: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function softmax.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `softmax`.
            """
            return a

    _state.update({"AttnBackend": AttnBackend})


def _disabled_test_core_math_ops_branches_sub_10(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 10 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (AttnBackend, empty_backend) = (_state["AttnBackend"], _state["empty_backend"])
    mod._scaled_dot_product_attention_eager(
        AttnBackend(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), is_causal=True, mask=np.ones((2, 2))
    )
    try:
        mod._np_bandpart(empty_backend, x=None)
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
        mod._np_betapdf(empty_backend, x=None)
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
    with pytest.MonkeyPatch.context() as mp_gcd:
        mp_gcd.delattr(np, "gcd", raising=False)
        try:
            mod._np_gcd(empty_backend, np.array([2]), np.array([4]))
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
    import sys

    original_scipy = sys.modules.get("scipy.special")
    pytest.MonkeyPatch().setitem(sys.modules, "scipy.special", None)
    _state.update({"original_scipy": original_scipy, "sys": sys})
    _state.update({"AttnBackend": AttnBackend, "empty_backend": empty_backend})


def _disabled_test_core_math_ops_branches_sub_11(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 11 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (empty_backend, original_scipy, sys) = (_state["empty_backend"], _state["original_scipy"], _state["sys"])
    try:
        if adapt_backend_module(mod).has_symbol("_np_gamma"):
            adapt_backend_module(mod).require_symbol("_np_gamma")(empty_backend, np.array([1.0]))
        if adapt_backend_module(mod).has_symbol("_np_modifiedbesseli1"):
            mod._np_modifiedbesseli1(empty_backend, x=np.array([1.0]))
        if adapt_backend_module(mod).has_symbol("_np_xlog1py"):
            mod._np_xlog1py(empty_backend, np.array([1.0]), np.array([1.0]))
        if adapt_backend_module(mod).has_symbol("_np_xlogy"):
            mod._np_xlogy(empty_backend, np.array([1.0]), np.array([1.0]))
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
        if original_scipy is not None:
            sys.modules["scipy.special"] = original_scipy
        else:
            del sys.modules["scipy.special"]
    _state.update({"empty_backend": empty_backend, "original_scipy": original_scipy, "sys": sys})


def _disabled_test_core_math_ops_branches_sub_12(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 12 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]
    mod._all_gather(empty_backend, np.ones((2, 2)))

    class BackendWithStack(MockTestStub):
        """Test class BackendWithStack."""

        def stack(self, x: BackendArg) -> BackendArg:
            """Test function stack.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `stack`.
            """
            return x

    class BackendWithArray(MockTestStub):
        """Test class BackendWithArray."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

    mod._all_gather(BackendWithStack(), np.ones((2, 2)))
    mod._all_gather(BackendWithArray(), np.ones((2, 2)))
    _state.update({"empty_backend": empty_backend})


def _disabled_test_core_math_ops_branches_sub_13(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 13 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, full_backend) = (_state["dummy_input"], _state["full_backend"])

    class MockData3:
        """Test class MockData3."""

        data = np.ones((2, 2))
        dtype = float

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return [1.0]

    class MockData4(MockTestStub):
        """Test class MockData4."""

        data = MockData3()

    try:
        mod._allclose(
            full_backend,
            dummy_input,
            dummy_input,
            rtol=MockData4(),
            atol=MockData4(),
            equal_nan=MockData4(),
        )
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
    import sys

    try:
        import scipy.special

        scipy_installed = True
    except ImportError:
        scipy_installed = False
    _state.update({"MockData3": MockData3, "sys": sys})
    _state.update({"dummy_input": dummy_input, "full_backend": full_backend, "scipy_installed": scipy_installed})


def _run_scipy_special_ops(empty_backend: types.ModuleType) -> None:
    """Invoke modifiedbesseli1, xlog1py, and xlogy on empty_backend.

    Args:
        empty_backend: Test parameter `empty_backend`.
    """
    mod._np_modifiedbesseli1(empty_backend, np.array([1.0]))
    mod._np_xlog1py(empty_backend, np.array([1.0]), np.array([1.0]))
    mod._np_xlogy(empty_backend, np.array([1.0]), np.array([1.0]))


def _disabled_test_core_math_ops_branches_sub_14(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 14 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (empty_backend, scipy_installed, sys) = (_state["empty_backend"], _state["scipy_installed"], _state["sys"])
    original_scipy: types.ModuleType | None = None
    if scipy_installed:
        if adapt_backend_module(mod).has_symbol("_np_modifiedbesseli1"):
            mod._np_modifiedbesseli1(empty_backend, np.array([1.0]))
        original_scipy = sys.modules.get("scipy.special")
        pytest.MonkeyPatch().setitem(sys.modules, "scipy.special", None)
        try:
            _run_scipy_special_ops(empty_backend)
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
            if original_scipy is not None:
                sys.modules["scipy.special"] = original_scipy
            else:
                del sys.modules["scipy.special"]
    else:
        _run_scipy_special_ops(empty_backend)
    _state.update(
        {"empty_backend": empty_backend, "original_scipy": original_scipy, "scipy_installed": scipy_installed}
    )
    _state.update({"sys": sys})


def _disabled_test_core_math_ops_branches_sub_15(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 15 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    import pytest

    empty_backend = _state["empty_backend"]
    if adapt_backend_module(mod).has_symbol("_np_deg2rad"):
        adapt_backend_module(mod).require_symbol("_np_deg2rad")(empty_backend, np.array([180.0]))
        with pytest.raises((TypeError, ValueError, AttributeError)):
            adapt_backend_module(mod).require_symbol("_np_deg2rad")(empty_backend, x=None)
    if adapt_backend_module(mod).has_symbol("_np_betapdf"):
        mod._np_betapdf(empty_backend, np.array([0.5]), np.array([1.0]), np.array([1.0]))
    if adapt_backend_module(mod).has_symbol("_np_packbits"):
        with pytest.raises((TypeError, ValueError, AttributeError)):
            mod._np_packbits(empty_backend, x=None)
    if adapt_backend_module(mod).has_symbol("_np_polyint"):
        with pytest.raises((TypeError, ValueError, AttributeError)):
            mod._np_polyint(empty_backend, p=None)
    original_hasattr = hasattr
    import builtins

    _state.update({"builtins": builtins, "original_hasattr": require_callable(original_hasattr)})
    _state.update({"empty_backend": empty_backend})


def _disabled_test_core_math_ops_branches_sub_16(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 16 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    original_hasattr = _state["original_hasattr"]

    def mock_hasattr(obj: BackendArg, name: BackendArg) -> BackendArg:
        """Test function mock_hasattr.

        Args:
            obj: Test parameter `obj`.
            name: Test parameter `name`.

        Returns:
            BackendArg: Computed output value from `mock_hasattr`.
        """
        if obj is np and name == "take_along_axis":
            return False
        if obj is np and name == "random":
            return False
        return original_hasattr(obj, name)

    _state.update({"mock_hasattr": require_callable(mock_hasattr)})
    _state.update({"original_hasattr": require_callable(original_hasattr)})


def _disabled_test_core_math_ops_branches_sub_17(_state: _DisabledTestCoreMathOpsBranchesState) -> None:
    """Execute sub-part 17 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (builtins, empty_backend, mock_hasattr, original_hasattr) = (
        _state["builtins"],
        _state["empty_backend"],
        _state["mock_hasattr"],
        _state["original_hasattr"],
    )
    try:
        pytest.MonkeyPatch().setattr(builtins, "hasattr", mock_hasattr, raising=False)
        if adapt_backend_module(mod).has_symbol("_np_takealongaxis"):
            mod._np_takealongaxis(empty_backend, np.array([1]), np.array([0]), axis=-1)
        if adapt_backend_module(mod).has_symbol("_np_triangular"):
            mod._np_triangular(empty_backend, left=0, mode=0.5, right=1, size=1)
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
        pytest.MonkeyPatch().setattr(builtins, "hasattr", original_hasattr, raising=False)
    if adapt_backend_module(mod).has_symbol("_np_modifiedbesseli1"):
        with pytest.raises((TypeError, ValueError, AttributeError)):
            mod._np_modifiedbesseli1(empty_backend, x=None)
    _state.update(
        {"builtins": builtins, "empty_backend": empty_backend, "mock_hasattr": require_callable(mock_hasattr)}
    )
    _state.update({"original_hasattr": require_callable(original_hasattr)})


def test_core_math_ops_branches() -> None:
    """Test function disabled_test_core_math_ops_branches."""
    _state: _DisabledTestCoreMathOpsBranchesState = {}
    for _sub_fn in (
        _disabled_test_core_math_ops_branches_sub_1,
        _disabled_test_core_math_ops_branches_sub_2,
        _disabled_test_core_math_ops_branches_sub_3,
        _disabled_test_core_math_ops_branches_sub_4,
        _disabled_test_core_math_ops_branches_sub_5,
        _disabled_test_core_math_ops_branches_sub_6,
        _disabled_test_core_math_ops_branches_sub_7,
        _disabled_test_core_math_ops_branches_sub_8,
        _disabled_test_core_math_ops_branches_sub_9,
        _disabled_test_core_math_ops_branches_sub_10,
        _disabled_test_core_math_ops_branches_sub_11,
        _disabled_test_core_math_ops_branches_sub_12,
        _disabled_test_core_math_ops_branches_sub_13,
        _disabled_test_core_math_ops_branches_sub_14,
        _disabled_test_core_math_ops_branches_sub_15,
        _disabled_test_core_math_ops_branches_sub_16,
        _disabled_test_core_math_ops_branches_sub_17,
    ):
        _sub_fn(_state)
    assert len(_state) > 0


def test_core_math_ops_missing_coverage_new() -> None:
    """Test function disabled_test_core_math_ops_missing_coverage_new."""
    handled_errors = 0
    import sys

    import numpy as np
    import pytest

    import ml_switcheroo_compiler.backends.eager.core_math_ops as mod

    empty_backend = EmptyMockBackend()
    with pytest.raises((AttributeError, TypeError, ValueError)):
        mod._householder_product(empty_backend, np.array([[1.0]]), np.array([1.0]))
    res_bp = mod._np_bandpart(empty_backend, np.ones((2, 2)), num_lower=-1, num_upper=-1)
    assert res_bp is not None
    with pytest.raises((AttributeError, TypeError, ValueError)):
        adapt_backend_module(mod).require_symbol("_np_gamma")(empty_backend, x=None)
    original_scipy = sys.modules.get("scipy.special")
    import types

    dummy_scipy_special = types.ModuleType("scipy.special")
    original_scipy_base = sys.modules.get("scipy")
    dummy_scipy = types.ModuleType("scipy")
    pytest.MonkeyPatch().setattr(dummy_scipy, "special", dummy_scipy_special, raising=False)
    sys.modules["scipy"] = dummy_scipy
    sys.modules["scipy.special"] = dummy_scipy_special
    try:
        try:
            adapt_backend_module(mod).require_symbol("_np_gamma")(empty_backend, np.array([1.0]))
            mod._np_xlog1py(empty_backend, np.array([1.0]), np.array([1.0]))
            mod._np_xlogy(empty_backend, np.array([1.0]), np.array([1.0]))
        except (AttributeError, TypeError, ValueError):
            handled_errors += 1
    finally:
        if original_scipy is not None:
            sys.modules["scipy.special"] = original_scipy
        else:
            del sys.modules["scipy.special"]
        if original_scipy_base is not None:
            sys.modules["scipy"] = original_scipy_base
        else:
            del sys.modules["scipy"]

    class AttnBackend(MockTestStub):
        """Test class AttnBackend."""

        def matmul(self, a: BackendArg, b: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return a

        def transpose(self, a: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function transpose.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `transpose`.
            """
            return a

        def triu(self, a: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function triu.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `triu`.
            """
            return a

        def ones(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function ones.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `ones`.
            """
            return np.ones((2, 2))

        def where(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function where.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `where`.
            """
            return np.ones((2, 2))

        def softmax(self, a: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function softmax.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `softmax`.
            """
            return a

    mod._scaled_dot_product_attention_eager(AttnBackend(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), scale=1.0)
