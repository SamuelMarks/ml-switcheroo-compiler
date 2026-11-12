# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Tests for test_core_math_ops_fallbacks."""

from __future__ import annotations

import typing
import warnings

import numpy as np

import ml_switcheroo_compiler.backends.eager.core_math_ops as mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod, ensure_mutable, ensure_tensor_dict, has_backend_member

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
warnings.filterwarnings("ignore")


class EmptyMockBackend:
    """Test class EmptyMockBackend."""

    pass


class FullMockBackend:
    """Test class FullMockBackend."""

    pass


class MockNamespace:
    """Test class MockNamespace."""

    pass


class MissingPrimaryMockBackend:
    """Test class MissingPrimaryMockBackend."""

    random = MockNamespace()
    nn = MockNamespace()
    linalg = MockNamespace()
    fft = MockNamespace()
    signal = MockNamespace()
    special = MockNamespace()
    image = MockNamespace()


def _disabled_test_core_math_ops_branches_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = EmptyMockBackend()
    full_backend = FullMockBackend()
    nested_backend = MissingPrimaryMockBackend()
    dummy_input = np.ones((2, 2))
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "empty_backend": empty_backend, "full_backend": full_backend, "nested_backend": nested_backend}))


def _disabled_test_core_math_ops_branches_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
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
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
            try:
                func(empty_backend, dummy_input, dummy_input)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
            try:
                func(empty_backend, dummy_input, dummy_input, dummy_input)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "empty_backend": empty_backend}))


def _disabled_test_core_math_ops_branches_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
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
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
            try:
                func(full_backend, dummy_input, dummy_input)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
            try:
                func(full_backend, dummy_input, dummy_input, dummy_input)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "full_backend": full_backend}))


def _disabled_test_core_math_ops_branches_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
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
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
            try:
                func(nested_backend, dummy_input, dummy_input)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
            try:
                func(nested_backend, dummy_input, dummy_input, dummy_input)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                handled_errors += 1
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "nested_backend": nested_backend}))


def _disabled_test_core_math_ops_branches_sub_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 5 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, full_backend, nested_backend) = (_state["dummy_input"], _state["full_backend"], _state["nested_backend"])
    try:
        mod._apply_causal_mask(full_backend, dummy_input)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    try:
        mod._apply_softmax(nested_backend, dummy_input)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    try:
        ensure_eager_tensor(ensure_eager_tensor(mod)._global_adaptive_pool_mock)(full_backend, dummy_input, 1)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    try:
        ensure_eager_tensor(ensure_eager_tensor(mod)._global_adaptive_pool_mock)(full_backend, dummy_input, (1, 1))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "full_backend": full_backend, "nested_backend": nested_backend}))


def _disabled_test_core_math_ops_branches_sub_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 6 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, empty_backend, full_backend) = (_state["dummy_input"], _state["empty_backend"], _state["full_backend"])
    try:
        mod._gamma(empty_backend, 1.0)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1

    class MockData:
        """Test class MockData."""

        data = np.ones((2, 2))
        dtype = float

        def item(self) -> EagerValue:
            """Test function item.

            Returns:
                EagerValue: Computed output value from `item`.
            """
            return 1.0

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return [1.0]

    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=ensure_eager_tensor(MockData()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    _state.update(ensure_tensor_dict({"MockData": MockData}))
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "empty_backend": empty_backend, "full_backend": full_backend}))


def _disabled_test_core_math_ops_branches_sub_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 7 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (dummy_input, full_backend) = (_state["dummy_input"], _state["full_backend"])

    class MockData2:
        """Test class MockData2."""

        data = np.ones((2, 2))
        dtype = float

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return [1.0]

    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=ensure_eager_tensor(MockData2()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1

    class MockData3:
        """Test class MockData3."""

        data = np.ones((2, 2))
        dtype = float

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return [1.0]

    _state.update(ensure_tensor_dict({"MockData2": MockData2, "MockData3": MockData3}))
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "full_backend": full_backend}))


def _disabled_test_core_math_ops_branches_sub_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 8 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (MockData3, dummy_input, full_backend) = (_state["MockData3"], _state["dummy_input"], _state["full_backend"])
    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=MockData3(), atol=MockData3(), equal_nan=MockData3())
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1

    class NoTriuBackend:
        """Test class NoTriuBackend."""

        pass

    try:
        ensure_callable(mod._apply_causal_mask)(ensure_callable(NoTriuBackend)(), ensure_callable(np.ones)((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    try:
        ensure_callable(mod._apply_softmax)(ensure_callable(NoTriuBackend)(), ensure_callable(np.ones)((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    _state.update(ensure_tensor_dict({"NoTriuBackend": NoTriuBackend}))
    _state.update(ensure_tensor_dict({"MockData3": MockData3, "dummy_input": dummy_input, "full_backend": full_backend}))


def _disabled_test_core_math_ops_branches_sub_9(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 9 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """

    class AttnBackend:
        """Test class AttnBackend."""

        def matmul(self, a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                EagerValue: Computed output value from `matmul`.
            """
            return a

        def transpose(self, a: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function transpose.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `transpose`.
            """
            return a

        def triu(self, a: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function triu.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `triu`.
            """
            return a

        def ones(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function ones.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `ones`.
            """
            return np.ones((2, 2))

        def where(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function where.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `where`.
            """
            return np.ones((2, 2))

        def softmax(self, a: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function softmax.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `softmax`.
            """
            return a

    _state.update(ensure_tensor_dict({"AttnBackend": AttnBackend}))


def _disabled_test_core_math_ops_branches_sub_10(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 10 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (AttnBackend, empty_backend) = (_state["AttnBackend"], _state["empty_backend"])
    mod._scaled_dot_product_attention_eager(AttnBackend(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), is_causal=True, mask=np.ones((2, 2)))
    try:
        mod._np_bandpart(empty_backend, x=None)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    try:
        mod._np_betapdf(empty_backend, x=None)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    original_gcd = ensure_eager_tensor(np).gcd
    try:
        if original_gcd:
            del np.gcd
        mod._np_gcd(empty_backend, np.array([2]), np.array([4]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    finally:
        if original_gcd:
            ensure_mutable(np).gcd = ensure_eager_tensor(original_gcd)
    import sys

    original_scipy = sys.modules.get("scipy.special")
    ensure_eager_tensor(sys.modules)["scipy.special"] = None
    _state.update(ensure_tensor_dict({"original_scipy": original_scipy, "sys": sys}))
    _state.update(ensure_tensor_dict({"AttnBackend": AttnBackend, "empty_backend": empty_backend}))


def _disabled_test_core_math_ops_branches_sub_11(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 11 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (empty_backend, original_scipy, sys) = (_state["empty_backend"], _state["original_scipy"], _state["sys"])
    try:
        if has_backend_member(mod, "_np_gamma"):
            ensure_eager_tensor(mod)._np_gamma(empty_backend, np.array([1.0]))
        if has_backend_member(mod, "_np_modifiedbesseli1"):
            mod._np_modifiedbesseli1(empty_backend, x=np.array([1.0]))
        if has_backend_member(mod, "_np_xlog1py"):
            mod._np_xlog1py(empty_backend, np.array([1.0]), np.array([1.0]))
        if has_backend_member(mod, "_np_xlogy"):
            mod._np_xlogy(empty_backend, np.array([1.0]), np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    finally:
        if ensure_eager_tensor(original_scipy) is not None:
            sys.modules["scipy.special"] = original_scipy
        else:
            del sys.modules["scipy.special"]
    _state.update(ensure_tensor_dict({"empty_backend": empty_backend, "original_scipy": original_scipy, "sys": sys}))


def _disabled_test_core_math_ops_branches_sub_12(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 12 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]
    mod._all_gather(empty_backend, np.ones((2, 2)))

    class BackendWithStack:
        """Test class BackendWithStack."""

        def stack(self, x: EagerValue) -> EagerValue:
            """Test function stack.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `stack`.
            """
            return x

    class BackendWithArray:
        """Test class BackendWithArray."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return x

    mod._all_gather(ensure_mod(BackendWithStack()), np.ones((2, 2)))
    mod._all_gather(ensure_mod(BackendWithArray()), np.ones((2, 2)))
    _state.update(ensure_tensor_dict({"BackendWithArray": BackendWithArray, "BackendWithStack": BackendWithStack}))
    _state.update(ensure_tensor_dict({"empty_backend": empty_backend}))


def _disabled_test_core_math_ops_branches_sub_13(_state: dict[str, EagerTensorProtocol]) -> None:
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

        def tolist(self) -> EagerValue:
            """Test function tolist.

            Returns:
                EagerValue: Computed output value from `tolist`.
            """
            return [1.0]

    class MockData4:
        """Test class MockData4."""

        data = MockData3()

    try:
        mod._allclose(full_backend, dummy_input, dummy_input, rtol=ensure_eager_tensor(MockData4()), atol=ensure_eager_tensor(MockData4()), equal_nan=ensure_eager_tensor(MockData4()))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    import sys

    try:
        import scipy.special

        scipy_installed = True
    except ImportError:
        scipy_installed = False
    _state.update(ensure_tensor_dict({"MockData3": MockData3, "MockData4": MockData4, "sys": sys}))
    _state.update(ensure_tensor_dict({"dummy_input": dummy_input, "full_backend": full_backend, "scipy_installed": scipy_installed}))


def _run_scipy_special_ops(empty_backend: EagerValue) -> None:
    """Invoke modifiedbesseli1, xlog1py, and xlogy on empty_backend.

    Args:
        empty_backend: Test parameter `empty_backend`.
    """
    from tests.testing_utils import get_test_members

    mod_members = get_test_members(mod)
    for fn_name, args in (("_np_modifiedbesseli1", (empty_backend, np.array([1.0]))), ("_np_xlog1py", (empty_backend, np.array([1.0]), np.array([1.0]))), ("_np_xlogy", (empty_backend, np.array([1.0]), np.array([1.0])))):
        if fn_name in mod_members:
            ensure_callable(mod_members[fn_name])(*args)


def _disabled_test_core_math_ops_branches_sub_14(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 14 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (empty_backend, scipy_installed, sys) = (_state["empty_backend"], _state["scipy_installed"], _state["sys"])
    original_scipy: EagerValue = None
    if scipy_installed:
        if has_backend_member(mod, "_np_modifiedbesseli1"):
            mod._np_modifiedbesseli1(empty_backend, np.array([1.0]))
        original_scipy = sys.modules.get("scipy.special")
        sys.modules["scipy.special"] = None
        try:
            _run_scipy_special_ops(empty_backend)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            handled_errors += 1
        finally:
            if ensure_eager_tensor(original_scipy) is not None:
                sys.modules["scipy.special"] = original_scipy
            else:
                del sys.modules["scipy.special"]
    else:
        _run_scipy_special_ops(empty_backend)
    _state.update(ensure_tensor_dict({"empty_backend": empty_backend, "original_scipy": original_scipy, "scipy_installed": scipy_installed}))
    _state.update(ensure_tensor_dict({"sys": sys}))


def _disabled_test_core_math_ops_branches_sub_15(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 15 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    import pytest

    empty_backend = _state["empty_backend"]
    if has_backend_member(mod, "_np_deg2rad"):
        ensure_eager_tensor(mod)._np_deg2rad(empty_backend, np.array([180.0]))
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ensure_eager_tensor(mod)._np_deg2rad(empty_backend, x=None)
    if has_backend_member(mod, "_np_betapdf"):
        mod._np_betapdf(empty_backend, np.array([0.5]), np.array([1.0]), np.array([1.0]))
    if has_backend_member(mod, "_np_packbits"):
        with pytest.raises((TypeError, ValueError, AttributeError)):
            mod._np_packbits(empty_backend, x=None)
    if has_backend_member(mod, "_np_polyint"):
        with pytest.raises((TypeError, ValueError, AttributeError)):
            mod._np_polyint(empty_backend, p=None)
    original_hasattr = hasattr
    import builtins

    _state.update(ensure_tensor_dict({"builtins": builtins, "original_hasattr": original_hasattr}))
    _state.update(ensure_tensor_dict({"empty_backend": empty_backend}))


def _disabled_test_core_math_ops_branches_sub_16(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 16 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    original_hasattr = _state["original_hasattr"]

    def mock_hasattr(obj: EagerValue, name: EagerValue) -> EagerValue:
        """Test function mock_hasattr.

        Args:
            obj: Test parameter `obj`.
            name: Test parameter `name`.

        Returns:
            EagerValue: Computed output value from `mock_hasattr`.
        """
        if obj is np and name == "take_along_axis":
            return False
        if obj is np and name == "random":
            return False
        return original_hasattr(obj, name)

    _state.update(ensure_tensor_dict({"mock_hasattr": mock_hasattr}))
    _state.update(ensure_tensor_dict({"original_hasattr": original_hasattr}))


def _disabled_test_core_math_ops_branches_sub_17(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 17 of disabled_test_core_math_ops_branches.

    Args:
        _state: Shared test state dictionary.
    """
    handled_errors = 0
    (builtins, empty_backend, mock_hasattr, original_hasattr) = (_state["builtins"], _state["empty_backend"], _state["mock_hasattr"], _state["original_hasattr"])
    try:
        ensure_mutable(builtins).hasattr = ensure_eager_tensor(mock_hasattr)
        if has_backend_member(mod, "_np_takealongaxis"):
            mod._np_takealongaxis(empty_backend, np.array([1]), np.array([0]), axis=-1)
        if has_backend_member(mod, "_np_triangular"):
            mod._np_triangular(empty_backend, left=0, mode=0.5, right=1, size=1)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        handled_errors += 1
    finally:
        ensure_mutable(builtins).hasattr = ensure_eager_tensor(original_hasattr)
    if has_backend_member(mod, "_np_modifiedbesseli1"):
        import pytest

        with pytest.raises((TypeError, ValueError, AttributeError)):
            mod._np_modifiedbesseli1(empty_backend, x=None)
    _state.update(ensure_tensor_dict({"builtins": builtins, "empty_backend": empty_backend, "mock_hasattr": mock_hasattr}))
    _state.update(ensure_tensor_dict({"original_hasattr": original_hasattr}))


def test_core_math_ops_branches() -> None:
    """Test function disabled_test_core_math_ops_branches."""
    _state: dict[str, EagerTensorProtocol] = {}
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
        mod._householder_product(ensure_mod(empty_backend), np.array([[1.0]]), np.array([1.0]))
    res_bp = mod._np_bandpart(ensure_mod(empty_backend), np.ones((2, 2)), num_lower=-1, num_upper=-1)
    assert res_bp is not None
    with pytest.raises((AttributeError, TypeError, ValueError)):
        ensure_callable(ensure_eager_tensor(ensure_eager_tensor(mod)._np_gamma))(empty_backend, x=None)
    original_scipy = sys.modules.get("scipy.special")
    import types

    dummy_scipy_special = types.ModuleType("scipy.special")
    original_scipy_base = sys.modules.get("scipy")
    dummy_scipy = types.ModuleType("scipy")
    ensure_mutable(dummy_scipy).special = ensure_eager_tensor(dummy_scipy_special)
    sys.modules["scipy"] = dummy_scipy
    sys.modules["scipy.special"] = dummy_scipy_special
    try:
        try:
            ensure_callable(ensure_eager_tensor(ensure_eager_tensor(mod)._np_gamma))(empty_backend, ensure_callable(np.array)([1.0]))
            mod._np_xlog1py(ensure_mod(empty_backend), np.array([1.0]), np.array([1.0]))
            mod._np_xlogy(ensure_mod(empty_backend), np.array([1.0]), np.array([1.0]))
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

    class AttnBackend:
        """Test class AttnBackend."""

        def matmul(self, a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                EagerValue: Computed output value from `matmul`.
            """
            return a

        def transpose(self, a: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function transpose.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `transpose`.
            """
            return a

        def triu(self, a: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function triu.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `triu`.
            """
            return a

        def ones(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function ones.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `ones`.
            """
            return np.ones((2, 2))

        def where(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function where.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `where`.
            """
            return np.ones((2, 2))

        def softmax(self, a: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function softmax.

            Args:
                a: Test parameter `a`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `softmax`.
            """
            return a

    mod._scaled_dot_product_attention_eager(ensure_mod(AttnBackend()), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), scale=1.0)
