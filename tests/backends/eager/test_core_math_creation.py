"""Tests for test_core_math_creation."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_creation import (
    _from_dlpack,
    _fromfunction,
    _fromiter,
    _frompyfunc,
    _geomspace,
    _mgrid,
    _np_fromfunction,
    _np_fromiter,
    _np_frompyfunc,
    _ogrid,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_creation_coverage() -> None:
    """Test 100% lines and branches of math_creation.

    Returns:
        None
    """

    class MockBackend:
        """Mock backend providing creation functions."""

        def fromfunction(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock fromfunction.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "fromfunction_res"

        def from_dlpack(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock from_dlpack.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "from_dlpack_res"

        def fromiter(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock fromiter.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "fromiter_res"

        def frompyfunc(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock frompyfunc.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "frompyfunc_res"

        def geomspace(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock geomspace.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "geomspace_res"

        def mgrid(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock mgrid.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "mgrid_res"

        def ogrid(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock ogrid.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "ogrid_res"

    backend = MockBackend()
    assert _fromfunction(as_mod(backend), lambda i: i, (3,)) == "fromfunction_res"
    assert _from_dlpack(as_mod(backend), None) == "from_dlpack_res"
    assert _fromiter(as_mod(backend), [1, 2], float) == "fromiter_res"
    assert _frompyfunc(as_mod(backend), lambda x: x, 1, 1) == "frompyfunc_res"
    assert _geomspace(as_mod(backend), 1, 1000, 4) == "geomspace_res"
    assert _mgrid(as_mod(backend), slice(0, 5)) == "mgrid_res"
    assert _ogrid(as_mod(backend), slice(0, 5)) == "ogrid_res"

    # Test _np_fromfunction, _np_fromiter, _np_frompyfunc
    # When backend has the function
    assert _np_fromfunction(as_mod(backend), lambda i: i, (3,)) == "fromfunction_res"
    assert _np_fromiter(as_mod(backend), [1, 2], float) == "fromiter_res"
    assert _np_frompyfunc(as_mod(backend), lambda x: x, 1, 1) == "frompyfunc_res"

    # When backend lacks the function -> fallbacks to numpy
    class EmptyBackend:
        """Empty backend lacking creation methods."""

        pass

    empty_backend = EmptyBackend()
    fn_res = _np_fromfunction(as_mod(empty_backend), lambda i, j: i + j, (2, 2), dtype=int)
    assert as_tensor(getattr(fn_res, "shape")) == (2, 2)

    it_res = _np_fromiter(as_mod(empty_backend), as_tensor(x * 2 for x in range(3)), dtype=int)
    assert as_callable(list)(it_res) == [0, 2, 4]

    pyf = _np_frompyfunc(as_mod(empty_backend), lambda x: x * 3, 1, 1)
    assert as_callable(pyf)(5) == 15

    # Registry verification
    for op_name in [
        "Fromfunction",
        "FromDlpack",
        "Fromiter",
        "Frompyfunc",
        "Geomspace",
        "Mgrid",
        "Ogrid",
        "NpFromfunction",
        "NpFromiter",
        "NpFrompyfunc",
    ]:
        assert global_eager_registry.get(op_name) is not None
