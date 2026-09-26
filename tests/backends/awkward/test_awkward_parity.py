"""Exhaustive unit and parity tests for Awkward Array backend supporting native ragged structures."""

from __future__ import annotations

import sys
from typing import Any
from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.awkward as ak_pkg
from ml_switcheroo_compiler.backends.awkward.eager import (
    _eval_operator,
    _eval_ragged_fallback,
    _unwrap_arg,
    execute_op,
)
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
from ml_switcheroo_compiler.backends.awkward.types import (
    broadcast_ragged,
    from_iter,
    from_numpy,
    from_regular,
    get_layout,
    is_ragged,
    ragged_shape,
    to_list,
    to_numpy,
    to_regular,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_jagged_reductions_without_padding() -> None:
    """Verify variable-length jagged reductions along axis -1 and full array without padding."""
    jagged_data = [[1, 2, 3], [4], [5, 6]]

    # Sum
    assert to_list(execute_op("Sum", jagged_data, axis=-1)) == [6, 4, 11]
    assert execute_op("Sum", jagged_data) == 21

    # Mean
    assert to_list(execute_op("Mean", jagged_data, axis=-1)) == [2.0, 4.0, 5.5]
    assert execute_op("Mean", jagged_data) == pytest.approx(3.5)

    # Min
    assert to_list(execute_op("Min", jagged_data, axis=-1)) == [1, 4, 5]
    assert execute_op("Min", jagged_data) == 1

    # Max
    assert to_list(execute_op("Max", jagged_data, axis=-1)) == [3, 4, 6]
    assert execute_op("Max", jagged_data) == 6

    # Prod
    assert to_list(execute_op("Prod", jagged_data, axis=-1)) == [6, 4, 30]
    assert execute_op("Prod", jagged_data) == 720

    # Count
    assert to_list(execute_op("Count", jagged_data, axis=-1)) == [3, 1, 2]
    assert execute_op("Count", jagged_data) == 6

    # Num and Flatten
    assert to_list(execute_op("Num", jagged_data)) == [3, 1, 2]
    assert to_list(execute_op("Flatten", jagged_data)) == [1, 2, 3, 4, 5, 6]


def test_jagged_arithmetic_and_operators() -> None:
    """Verify arithmetic operators on scalar and array operands."""
    # Binary math
    assert execute_op("Add", 10, 5) == 15
    assert execute_op("Sub", 10, 5) == 5
    assert execute_op("Mul", 10, 5) == 50
    assert execute_op("Div", 10, 5) == 2.0
    assert execute_op("FloorDivide", 11, 5) == 2
    assert execute_op("Mod", 11, 5) == 1
    assert execute_op("Pow", 2, 3) == 8

    # Comparison
    assert execute_op("Equal", 5, 5) is True
    assert execute_op("NotEqual", 5, 4) is True
    assert execute_op("Greater", 6, 5) is True
    assert execute_op("GreaterEqual", 5, 5) is True
    assert execute_op("Less", 4, 5) is True
    assert execute_op("LessEqual", 5, 5) is True

    # Logical and bitwise
    assert execute_op("LogicalAnd", True, False) is False
    assert execute_op("LogicalOr", True, False) is True
    assert execute_op("LogicalXor", True, False) is True
    assert execute_op("BitwiseAnd", 6, 3) == 2
    assert execute_op("BitwiseOr", 6, 3) == 7
    assert execute_op("BitwiseXor", 6, 3) == 5

    # Unary
    assert execute_op("Neg", 10) == -10
    assert execute_op("Abs", -10) == 10
    assert execute_op("Pos", 10) == 10
    assert execute_op("Invert", 0) == -1
    assert execute_op("LogicalNot", False) is True


def test_ragged_shape_and_broadcasting() -> None:
    """Verify ragged shape inference and broadcasting without padding."""
    jagged = [[1, 2, 3], [4], [5, 6]]
    regular = [[1, 2], [3, 4]]

    # Shape inference
    assert ragged_shape(jagged) == (3, None)
    assert ragged_shape(regular) == (2, 2)
    assert ragged_shape([]) == (0,)
    assert ragged_shape(42) == ()

    # is_ragged predicate
    assert is_ragged(jagged) is True
    assert is_ragged(regular) is False

    # Broadcasting scalar to ragged structure
    b_a, b_b = broadcast_ragged(10, jagged)
    assert to_list(b_a) == [[10, 10, 10], [10], [10, 10]]
    assert to_list(b_b) == jagged

    b_rev_a, b_rev_b = broadcast_ragged(jagged, 20)
    assert to_list(b_rev_a) == jagged
    assert to_list(b_rev_b) == [[20, 20, 20], [20], [20, 20]]

    # Classmethod invocation
    res_pair = AwkwardGenerator.broadcast_ragged(5, [[1], [2, 3]])
    assert to_list(res_pair[0]) == [[5], [5, 5]]


def test_layout_conversions() -> None:
    """Verify layout conversions between ragged, regular, numpy, and list formats."""
    jagged = [[1, 2], [3]]

    # to_regular padding
    regular_padded = to_regular(jagged, fill_value=0)
    np.testing.assert_array_equal(regular_padded, np.array([[1, 2], [3, 0]]))

    # from_regular
    from_reg = from_regular([[1, 2], [3, 4]])
    assert from_reg is not None

    # to_numpy and from_numpy
    np_arr = to_numpy([[1, 2], [3, 4]])
    assert isinstance(np_arr, np.ndarray)
    ak_from_np = from_numpy(np_arr)
    assert ak_from_np is not None

    # to_list and from_iter
    nested_list = to_list(np.array([1, 2, 3]))
    assert nested_list == [1, 2, 3]
    iter_arr = from_iter(range(3))
    assert iter_arr is not None

    # get_layout
    layout_info = get_layout(jagged)
    assert isinstance(layout_info, dict)
    assert layout_info["is_ragged"] is True

    # Classmethods on AwkwardGenerator
    assert AwkwardGenerator.is_ragged([[1], [2, 3]]) is True
    assert AwkwardGenerator.ragged_shape([[1], [2]]) == (2, 1)
    assert AwkwardGenerator.to_regular([[1, 2], [3]]) is not None
    assert AwkwardGenerator.from_regular([[1, 2]]) is not None
    assert AwkwardGenerator.to_numpy(np.array([1])) is not None
    assert AwkwardGenerator.from_numpy(np.array([1])) is not None
    assert AwkwardGenerator.to_list([1, 2]) == [1, 2]
    assert AwkwardGenerator.from_iter([1, 2]) is not None
    assert AwkwardGenerator.get_layout([[1]]) is not None


def test_generator_dedicated_visitors() -> None:
    """Verify Awkward code generator emits dedicated native AST transformations."""
    g = IRGraph()
    gen = AwkwardGenerator(g, ragged_mode=True)
    dummy_node = IRNode(id="n", op_type="Flatten")

    assert gen.visit_Flatten(dummy_node, ["x"]) == "ak.flatten(x)"
    assert gen.visit_Unflatten(dummy_node, ["x", "counts"]) == "ak.unflatten(x, counts)"
    assert gen.visit_Num(dummy_node, ["x"]) == "ak.num(x)"
    assert gen.visit_PadNone(dummy_node, ["x", "target"]) == "ak.pad_none(x, target)"
    assert gen.visit_FillNone(dummy_node, ["x", "val"]) == "ak.fill_none(x, val)"
    assert gen.visit_DropNone(dummy_node, ["x"]) == "ak.drop_none(x)"
    assert gen.visit_Zip(dummy_node, ["x", "y"]) == "ak.zip(x, y)"
    assert gen.visit_Unzip(dummy_node, ["rec"]) == "ak.unzip(rec)"
    assert gen.visit_WithField(dummy_node, ["rec", "f", "'field'"]) == "ak.with_field(rec, f, 'field')"
    assert gen.visit_Cartesian(dummy_node, ["x", "y"]) == "ak.cartesian(x, y)"
    assert gen.visit_Combinations(dummy_node, ["x", "2"]) == "ak.combinations(x, 2)"
    assert gen.visit_BroadcastArrays(dummy_node, ["x", "y"]) == "ak.broadcast_arrays(x, y)"
    assert gen.visit_ToRegular(dummy_node, ["x"]) == "ak.to_regular(x)"
    assert gen.visit_FromRegular(dummy_node, ["x"]) == "ak.from_regular(x)"


def test_eager_dispatch_error_and_fallbacks() -> None:
    """Verify error handling, Tensor unwrapping, and fallback paths in eager execution."""
    cfg = TensorConfig((2,), "float32", "cpu")
    t1 = Tensor(np.array([1.0, 2.0]), cfg)
    t2 = Tensor(np.array([3.0, 4.0]), cfg)

    # Tensor unwrapping
    assert _unwrap_arg(t1) is not None
    assert _unwrap_arg(5) == 5
    res = execute_op("Add", t1, t2)
    np.testing.assert_allclose(res, [4.0, 6.0])

    # Classmethod dispatch
    assert ak_pkg.execute_op("Add", 1, 2) == 3
    assert AwkwardGenerator.execute_op("Add", 1, 2) == 3

    # Global eager registry fallback
    def mock_ak_op(backend_ctx: Any, *args: Any, **kwargs: Any) -> str:
        del backend_ctx, args, kwargs
        return "custom_awkward_eval"

    with mock.patch.dict(global_eager_registry._registry, {"CustomAwkwardOp": mock_ak_op}):
        assert execute_op("CustomAwkwardOp", 1) == "custom_awkward_eval"

    # Unsupported op
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        execute_op("CompletelyUnknownAwkwardOp")

    # Mock awkward failure in ak_mod
    class BrokenAkMod:
        @staticmethod
        def sum(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError("internal awkward crash")

    with mock.patch.dict(sys.modules, {"awkward": BrokenAkMod}):
        with pytest.raises(BackendNotSupportedError, match="Failed executing awkward.sum"):
            execute_op("Sum", [1, 2])

    # Mock ufunc failure in numpy
    with mock.patch("numpy.sin", side_effect=RuntimeError("numpy ufunc error")):
        with pytest.raises(BackendNotSupportedError, match="Failed executing NumPy ufunc"):
            execute_op("Sin", [1.0])

    # Unhandled operator in _eval_operator
    handled, _ = _eval_operator("unknown_op_xyz", (1, 2))
    assert handled is False

    # Unhandled operator in _eval_ragged_fallback
    handled_ragged, _ = _eval_ragged_fallback("unknown_op_xyz", (1,), {})
    assert handled_ragged is False
    empty_ragged, _ = _eval_ragged_fallback("sum", (), {})
    assert empty_ragged is False


def test_awkward_fallbacks_and_comprehensive_coverage() -> None:
    """Verify fallback implementations when awkward functions raise or are unavailable."""
    # 1. Generator ragged_mode=False
    g = IRGraph()
    gen_non_ragged = AwkwardGenerator(g, ragged_mode=False)
    code_non_ragged = gen_non_ragged.generate()
    assert "# Native ragged layout mode enabled" not in code_non_ragged

    # 2. Direct _eval_ragged_fallback coverage
    assert _eval_ragged_fallback("flatten", ([[1, 2], 3],), {})[1] == [1, 2, 3]
    assert _eval_ragged_fallback("num", ([[1, 2], [3]],), {})[1] == [2, 1]
    assert _eval_ragged_fallback("count", ([[1, None], [2]],), {"axis": 1})[1] == [1, 1]
    assert _eval_ragged_fallback("count", ([[1, None], [2]],), {})[1] == 2

    assert _eval_ragged_fallback("sum", ([[1, 2], 3],), {"axis": 1})[1] == [3, 3]
    assert _eval_ragged_fallback("sum", ([[1, 2], 3],), {})[1] == 6

    assert _eval_ragged_fallback("mean", ([[1, 3], []],), {"axis": 1})[1] == [2.0, 0.0]
    assert _eval_ragged_fallback("mean", ([[1, 3], []],), {})[1] == 2.0
    assert _eval_ragged_fallback("mean", ([],), {})[1] == 0.0

    assert _eval_ragged_fallback("min", ([[1, 3], []],), {"axis": 1})[1] == [1, []]
    assert _eval_ragged_fallback("min", ([[1, 3]],), {})[1] == 1
    assert _eval_ragged_fallback("min", ([],), {})[1] is None

    assert _eval_ragged_fallback("max", ([[1, 3], []],), {"axis": 1})[1] == [3, []]
    assert _eval_ragged_fallback("max", ([[1, 3]],), {})[1] == 3
    assert _eval_ragged_fallback("max", ([],), {})[1] is None

    assert _eval_ragged_fallback("prod", ([[2, 3], 4],), {"axis": 1})[1] == [6, 4]
    assert _eval_ragged_fallback("prod", ([[2, 3], 4],), {})[1] == 24
    assert _eval_ragged_fallback("prod", ([],), {})[1] == 1

    # 3. broadcast_ragged fallback when awkward has no broadcast_arrays
    class FakeNoBroadcastAk:
        pass

    with mock.patch("ml_switcheroo_compiler.backends.awkward.types._get_ak_module", return_value=FakeNoBroadcastAk):
        b1, b2 = broadcast_ragged(10, [[1, 2], [3]])
        assert b1 == [[10, 10], [10]]
        assert b2 == [[1, 2], [3]]

        b3, b4 = broadcast_ragged([[1, 2], [3]], 20)
        assert b3 == [[1, 2], [3]]
        assert b4 == [[20, 20], [20]]

        b5, b6 = broadcast_ragged([[1]], [[2]])
        assert b5 == [[1]]
        assert b6 == [[2]]

    # 4. to_regular fallback when ak has no to_regular or raises
    class FakeNoToRegularAk:
        @staticmethod
        def to_regular(_: Any) -> Any:
            raise RuntimeError("to_regular error")

    with mock.patch("ml_switcheroo_compiler.backends.awkward.types._get_ak_module", return_value=FakeNoToRegularAk):
        res_reg = to_regular([[1, 2], 3], fill_value=9)
        assert res_reg.shape == (2, 2)
        assert res_reg[1, 1] == 9

        res_non_list = to_regular(np.array([1, 2]))
        assert len(res_non_list) == 2

    # 5. from_regular, to_numpy, from_numpy, from_iter fallbacks
    class FakeBrokenConversionsAk:
        @staticmethod
        def from_regular(_: Any) -> Any:
            raise RuntimeError("from_regular error")

        @staticmethod
        def to_numpy(_: Any) -> Any:
            raise RuntimeError("to_numpy error")

        @staticmethod
        def from_numpy(_: Any) -> Any:
            raise RuntimeError("from_numpy error")

        @staticmethod
        def from_iter(_: Any) -> Any:
            raise RuntimeError("from_iter error")

    with mock.patch("ml_switcheroo_compiler.backends.awkward.types._get_ak_module", return_value=FakeBrokenConversionsAk):
        assert from_regular([[1, 2]]) is not None
        assert to_numpy([[1, 2]]) is not None
        assert from_numpy(np.array([1, 2])) is not None
        assert from_iter([1, 2]) is not None

    # 6. to_list with tolist method
    class ObjectWithToList:
        def tolist(self) -> list[int]:
            return [10, 20]

    assert to_list(ObjectWithToList()) == [10, 20]

    # 7. get_layout with .layout attribute
    class ObjectWithLayout:
        layout = "dummy_layout"

    assert get_layout(ObjectWithLayout()) == "dummy_layout"


def test_awkward_full_branches_and_edge_cases() -> None:
    """Verify missing branch edge cases in eager dispatch and type conversions."""
    # 1. eager.py: lines 297-298 (ImportError on awkward import) and line 327 (ragged fallback return)
    real_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "awkward":
            raise ImportError("no awkward")
        return real_import(name, *args, **kwargs)

    with mock.patch("importlib.import_module", side_effect=fake_import):
        res = execute_op("Sum", [[1, 2], [3]], axis=1)
        assert res == [3, 3]

    # 2. eager.py: line 303->312 (ak_mod lacks target_fn_name in AK_OPS_MAP)
    class PartialAkMod:
        pass

    with mock.patch.dict(sys.modules, {"awkward": PartialAkMod}):
        # 'Sum' is in AK_OPS_MAP, but PartialAkMod does not have 'sum'
        res_fallback = execute_op("Sum", [[1, 2], [3]], axis=1)
        assert res_fallback == [3, 3]

    # 3. types.py: line 326 (ragged_shape with hasattr(shape))
    assert ragged_shape(np.zeros((2, 3))) == (2, 3)

    # 4. types.py: line 374 (to_regular on regular awkward/numpy array)
    ak_arr = ak_pkg.array([[1, 2], [3, 4]])
    assert to_regular(ak_arr) is not None

    # 5. types.py: lines 409, 429, 450, 490 (successful conversions on awkward module)
    np_mat = np.array([[1, 2], [3, 4]])
    assert from_regular(np_mat) is not None
    assert to_numpy(ak_arr) is not None
    assert from_numpy(np_mat) is not None
    assert from_iter([10, 20]) is not None

    # 6. types.py: line 475 (to_list on scalar data without list/tolist)
    assert to_list(42) == [42]

    # 7. types.py: lines 301-302 (broadcast_arrays raises Exception)
    mock_ak_bc_err = mock.MagicMock()
    mock_ak_bc_err.broadcast_arrays.side_effect = RuntimeError("broadcast error")
    with mock.patch("ml_switcheroo_compiler.backends.awkward.types._get_ak_module", return_value=mock_ak_bc_err):
        b_err_a, b_err_b = broadcast_ragged(10, [[1, 2], [3]])
        assert b_err_a == [[10, 10], [10]]
        assert b_err_b == [[1, 2], [3]]

    # 8. types.py: hasattr False branches (to_regular, from_regular, to_numpy, from_numpy, from_iter)
    class BareModule:
        pass

    with mock.patch("ml_switcheroo_compiler.backends.awkward.types._get_ak_module", return_value=BareModule):
        assert to_regular([[1, 2], [3]]) is not None
        assert from_regular([[1, 2]]) is not None
        assert to_numpy([[1, 2]]) is not None
        assert from_numpy(np.array([1, 2])) is not None
        assert from_iter([1, 2]) is not None
