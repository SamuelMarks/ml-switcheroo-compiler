"""Tests for test_core_math_creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_math_creation_coverage() -> None:
    """Test 100% lines and branches of math_creation.

    Returns:
        None
    """

    class MockBackend:
        """Mock backend providing creation functions."""

        def fromfunction(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock fromfunction."""
            return "fromfunction_res"

        def from_dlpack(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock from_dlpack."""
            return "from_dlpack_res"

        def fromiter(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock fromiter."""
            return "fromiter_res"

        def frompyfunc(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock frompyfunc."""
            return "frompyfunc_res"

        def geomspace(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock geomspace."""
            return "geomspace_res"

        def mgrid(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock mgrid."""
            return "mgrid_res"

        def ogrid(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock ogrid."""
            return "ogrid_res"

    backend = MockBackend()
    assert _fromfunction(backend, lambda i: i, (3,)) == "fromfunction_res"
    assert _from_dlpack(backend, None) == "from_dlpack_res"
    assert _fromiter(backend, [1, 2], float) == "fromiter_res"
    assert _frompyfunc(backend, lambda x: x, 1, 1) == "frompyfunc_res"
    assert _geomspace(backend, 1, 1000, 4) == "geomspace_res"
    assert _mgrid(backend, slice(0, 5)) == "mgrid_res"
    assert _ogrid(backend, slice(0, 5)) == "ogrid_res"

    # Test _np_fromfunction, _np_fromiter, _np_frompyfunc
    # When backend has the function
    assert _np_fromfunction(backend, lambda i: i, (3,)) == "fromfunction_res"
    assert _np_fromiter(backend, [1, 2], float) == "fromiter_res"
    assert _np_frompyfunc(backend, lambda x: x, 1, 1) == "frompyfunc_res"

    # When backend lacks the function -> fallbacks to numpy
    class EmptyBackend:
        """Empty backend lacking creation methods."""

        pass

    empty_backend = EmptyBackend()
    fn_res = _np_fromfunction(empty_backend, lambda i, j: i + j, (2, 2), dtype=int)
    assert cast("EagerTensorProtocol", getattr(fn_res, "shape")) == (2, 2)

    it_res = _np_fromiter(empty_backend, (x * 2 for x in range(3)), dtype=int)
    assert cast("Callable[..., EagerTensorProtocol]", list)(it_res) == [0, 2, 4]

    pyf = _np_frompyfunc(empty_backend, lambda x: x * 3, 1, 1)
    assert cast("Callable[..., EagerTensorProtocol]", pyf)(5) == 15

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
