"""Unit tests for math_internal module advanced math and configuration functions."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_internal as mi
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


class MockBackend:
    """Mock backend module for testing math_internal functions."""

    def get_printoptions(self) -> dict[str, int]:
        """Mock get_printoptions."""
        return {"precision": 8}

    def rawmatmul(self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]], b: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Mock rawmatmul.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", (typing.cast("EagerTensorProtocol", a) @ typing.cast("EagerTensorProtocol", b)) + typing.cast("EagerTensorProtocol", 1))


def _test_math_internal_operations_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = MockBackend()

    # 1. _np_population_count
    pop = mi._np_population_count(typing.cast("BackendModuleProtocol", bk), [0, 1, 3, 7, 15])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(pop, [0, 1, 2, 3, 4])

    # 2. _np_get_printoptions_
    assert mi._np_get_printoptions_(typing.cast("BackendModuleProtocol", bk)) == {"precision": 8}

    # 3. _np_affineconfig
    assert mi._np_affineconfig(typing.cast("BackendModuleProtocol", bk), scale=2.0) == {"scale": 2.0}

    # 4. _np_blurconfig
    blur = mi._np_blurconfig(typing.cast("BackendModuleProtocol", bk), kernel_size=(3, 3), sigma=1.0)
    assert blur is not None

    # 5. _np_customroot (with and without solve)
    res_root = mi._np_customroot(typing.cast("BackendModuleProtocol", bk), lambda x: x, 5.0, lambda f, g: 42.0)
    assert res_root == 42.0
    res_root_no_solve = mi._np_customroot(typing.cast("BackendModuleProtocol", bk), lambda x: x, 5.0)
    assert res_root_no_solve == 5.0

    # 6. _np_elasticconfig
    elastic = mi._np_elasticconfig(typing.cast("BackendModuleProtocol", bk), interpolation="nearest")
    assert elastic is not None

    # 7. LinearOperator configurations
    assert mi._np_linearoperator(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatoradjoint(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorcirculant(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorcirculant2d(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorcirculant3d(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorcomposition(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorhouseholder(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorkronecker(typing.cast("BackendModuleProtocol", bk)) is not None
    assert mi._np_linearoperatorlowrankupdate(typing.cast("BackendModuleProtocol", bk)) is not None
    getattr(_state, "update")({"bk": bk, "blur": blur, "elastic": elastic, "pop": pop})
    getattr(_state, "update")({"res_root": res_root, "res_root_no_solve": res_root_no_solve})
    _state.update(locals())


def _test_math_internal_operations_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    assert mi._np_linearoperatortoeplitz(bk) is not None

    # 8. _np_perspectiveconfig
    persp = mi._np_perspectiveconfig(bk, interpolation="nearest")
    assert persp is not None
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 0.0], [1.0, 2.0]])
    res_bkm = mi._np_rawmatmul(bk, a, b)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_bkm, a @ b + 1)

    class EmptyBackend:
        """Test class EmptyBackend."""

        pass

    res_default_mm = mi._np_rawmatmul(typing.cast("BackendModuleProtocol", EmptyBackend()), a, b)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_default_mm, a @ b)

    # 10. _np_rawmerge
    merge_list = mi._np_rawmerge(bk, [10, 20])
    assert typing.cast("EagerTensorProtocol", getattr(merge_list, "__getitem__")(0)) == 10
    assert typing.cast("EagerTensorProtocol", getattr(merge_list, "__getitem__")(1)) == 0

    merge_args = mi._np_rawmerge(bk, 10, 20)
    assert typing.cast("EagerTensorProtocol", getattr(merge_args, "__getitem__")(0)) == 10

    merge_empty = mi._np_rawmerge(bk)
    assert typing.cast("EagerTensorProtocol", getattr(merge_empty, "__getitem__")(0)) is None
    assert typing.cast("EagerTensorProtocol", getattr(merge_empty, "__getitem__")(1)) == -1

    # 11. _np_rawop
    assert mi._np_rawop(bk, 999) == 999
    assert mi._np_rawop(bk) is None
    getattr(_state, "update")({"EmptyBackend": EmptyBackend, "a": a, "b": b, "merge_args": merge_args})
    getattr(_state, "update")({"merge_empty": merge_empty, "merge_list": merge_list, "persp": persp, "res_bkm": res_bkm})
    getattr(_state, "update")({"res_default_mm": res_default_mm})
    _state.update(locals())


def _test_math_internal_operations_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    sw_true = mi._np_rawswitch(bk, "data", True)
    assert sw_true == (None, "data")
    sw_false = mi._np_rawswitch(bk, "data", False)
    assert sw_false == ("data", None)

    sw_op_true = mi._np_switchop(bk, "data", pred=True)
    assert sw_op_true == (None, "data")
    sw_op_false = mi._np_switchop(bk, "data", pred=False)
    assert sw_op_false == ("data", None)

    # 13. _np_scanop (invalid fn, empty elems, with acc, without acc)
    scan_invalid = mi._np_scanop(bk, "not_callable")
    assert scan_invalid == "not_callable"

    scan_empty = mi._np_scanop(bk, elems=np.array([]), fn=lambda x, y: x + y)
    assert typing.cast("EagerTensorProtocol", getattr(scan_empty, "size")) == 0

    scan_no_acc = mi._np_scanop(bk, elems=np.array([1, 2, 3]), fn=lambda x, y: x + y)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(scan_no_acc, [1, 3, 6])

    scan_with_acc = mi._np_scanop(bk, lambda x, y: x + y, np.array([1, 2, 3]), 10)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(scan_with_acc, [11, 13, 16])

    # 14. _np_tensorarrayread and _np_tensorarraywrite (both within bounds and extending)
    handle = [10, 20]
    assert mi._np_tensorarrayread(bk, handle, 1) == 20

    h_in = mi._np_tensorarraywrite(bk, handle, 0, 99)
    assert typing.cast("EagerTensorProtocol", getattr(h_in, "__getitem__")(0)) == 99

    h_ext = mi._np_tensorarraywrite(bk, handle, 4, 88)
    assert typing.cast("EagerCallableProtocol", len)(h_ext) == 5
    getattr(_state, "update")({"h_ext": h_ext, "h_in": h_in, "handle": handle, "scan_empty": scan_empty})
    getattr(_state, "update")({"scan_invalid": scan_invalid, "scan_no_acc": scan_no_acc, "scan_with_acc": scan_with_acc, "sw_false": sw_false})
    getattr(_state, "update")({"sw_op_false": sw_op_false, "sw_op_true": sw_op_true, "sw_true": sw_true})
    _state.update(locals())


def _test_math_internal_operations_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_internal_operations.

    Args:
        _state: Shared test state dictionary.
    """
    bk, h_ext = _state["bk"], _state["h_ext"]
    assert h_ext[4] == 88

    # 15. _np_tensorconfig
    tcfg = mi._np_tensorconfig(bk, shape=(2, 3), dtype="float32", device="cpu")
    assert tcfg is not None
    getattr(_state, "update")({"tcfg": tcfg})


def test_math_internal_operations() -> None:
    """Verify operations and configuration functions in math_internal.py."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_internal_operations_part_1(_state)
    _test_math_internal_operations_part_2(_state)
    _test_math_internal_operations_part_3(_state)
    _test_math_internal_operations_part_4(_state)
    assert len(_state) > 0
