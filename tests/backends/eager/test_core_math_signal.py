"""Tests for test_core_math_signal."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_signal import (
    _correlate,
    _np_windowhann,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from tests.testing_utils import as_callable, as_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_signal_coverage() -> None:
    """Test 100% lines and branches of math_signal.

    Returns:
        None
    """

    # _correlate
    # Case A: backend has correlate
    class CorrelateBackend:
        """Backend with correlate."""

        def correlate(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock correlate.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "correlate_val"

    assert _correlate(as_mod(CorrelateBackend()), [1, 2], [1]) == "correlate_val"

    # Case B: backend lacks correlate -> falls back to np.correlate
    class EmptyBackend:
        """Empty backend."""

        pass

    eb = EmptyBackend()
    res1 = _correlate(as_mod(eb), [1, 2, 3], [0, 1])
    assert as_callable(len)(res1) == 2

    # Case with kwargs dictionary containing mode
    res2 = _correlate(as_mod(eb), [1, 2, 3], [0, 1], mode="full")
    assert as_callable(len)(res2) == 4

    # Case where kwargs does not have get
    class NoGetDict:
        """Object without get."""

        pass

    # Pass non-dict or kwargs without get by unpacking invalid type or passing kwargs
    res3 = _correlate(as_mod(eb), [1, 2, 3], [0, 1], **{})
    assert as_callable(len)(res3) == 2

    # _np_windowhann
    # Case A: backend has windowhann
    class WindowhannBackend:
        """Backend with windowhann."""

        def windowhann(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock windowhann.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "windowhann_val"

    assert _np_windowhann(as_mod(WindowhannBackend()), 5) == "windowhann_val"

    # Case B: backend lacks windowhann -> fallback to np.hanning
    class EmptyBackend2:
        """Empty backend."""

        pass

    hann_res = _np_windowhann(as_mod(EmptyBackend2()), 5)
    assert as_callable(len)(hann_res) == 5

    # Registry
    assert global_eager_registry.get("Correlate") is not None
    assert global_eager_registry.get("WindowHann") is not None
