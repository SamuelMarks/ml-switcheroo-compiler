# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Test universal ForeignCall Operation."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable
from pathlib import Path

import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerCallableProtocol, EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ops.foreign_call import ForeignCall
from tests.testing_utils import EagerTensorProtocol, MockTestStub, adapt_backend_module


def test_foreign_call_introspection() -> None:
    """Test dynamic shape introspection."""

    def my_complex_func(x: BackendArg) -> tuple[int, int, int]:
        """Mock complex func.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[int, int, int]: Computed output value from `my_complex_func`.
        """
        return (1, 2, 3)

    fc = ForeignCall()

    # Test typing hints
    assert fc.infer_shape(my_complex_func) == (int, int, int)

    # Test kwargs
    assert fc.infer_shape(my_complex_func, output_shape=(4, 5)) == (4, 5)

    # Test AST parsing
    def ast_func() -> BackendArg:
        """Mock ast func.

        Returns:
            BackendArg: Computed output value from `ast_func`.
        """
        return (10, 20)

    assert fc.infer_shape(ast_func) == (10, 20)

    # Test dictionary / named tuple structure
    class ComplexCallable(MockTestStub):
        """Mock."""

        def __call__(self) -> BackendArg:
            """Call.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return {"a": (2, 3), "b": (4, 5)}

    assert fc.infer_shape(ComplexCallable()) == ()

    # Test obj with shapes
    class WithShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.shape = (5, 5)

    assert fc.infer_shape(WithShape()) == (5, 5)
    assert fc.infer_shape(None, WithShape()) == (5, 5)

    class WithOutputShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = (6, 6)

    assert fc.infer_shape(WithOutputShape()) == (6, 6)

    class WithOutputShapes:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = [(7, 7), (8, 8)]

    assert fc.infer_shape(WithOutputShapes()) == (7, 7)


class _TestForeignCallIntrospectionBranchesState(typing.TypedDict, total=False):
    """Typed shared state for `_test_foreign_call_introspection_branches` split helpers."""

    MockSubNode: type
    fc: EagerTensorProtocol | OpValue


def _test_foreign_call_introspection_branches_part_1(_state: _TestForeignCallIntrospectionBranchesState) -> None:
    """Execute part 1 of test_foreign_call_introspection_branches.

    Args:
        _state: Shared test state dictionary.
    """
    fc = ForeignCall()

    # args tuple unpacking check
    assert fc.infer_shape(None, type("Mock", (), {"shape": (9, 9)})()) == (9, 9)

    # empty output_shapes
    class EmptyOutputShapes(MockTestStub):
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes: list[BackendArg] = []

    assert fc.infer_shape(EmptyOutputShapes()) == ()

    # bad output_shape
    class BadOutputShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = 5

    assert fc.infer_shape(BadOutputShape()) == (5,)

    # subgraph handling
    class MockSubNode:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.shape_metadata = (11, 11)

    _state.update({"MockSubNode": MockSubNode, "fc": fc})


def _test_foreign_call_introspection_branches_part_2(_state: _TestForeignCallIntrospectionBranchesState) -> None:
    """Execute part 2 of test_foreign_call_introspection_branches.

    Args:
        _state: Shared test state dictionary.
    """
    MockSubNode, fc = _state["MockSubNode"], _state["fc"]

    class MockSubgraph:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.outputs = ["out1"]
            self.nodes = {"out1": MockSubNode()}

    class WithSubgraph:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.subgraph = MockSubgraph()

    assert adapt_backend_module(fc).require_symbol("infer_shape")(WithSubgraph()) == (11, 11)

    # typing.get_type_hints failure fallback
    def failing_typing_hints() -> None:
        """Mock."""
        pass

    failing_typing_hints.__annotations__ = {"return": "invalid_type_string_that_typing_cannot_resolve"}
    assert adapt_backend_module(fc).require_symbol("infer_shape")(failing_typing_hints) == ()

    # callable without any return hints
    def empty_callable() -> None:
        """Mock."""
        pass

    assert adapt_backend_module(fc).require_symbol("infer_shape")(empty_callable) == ()
    _state.update({"MockSubNode": MockSubNode, "fc": fc})


def _test_foreign_call_introspection_branches_part_3(_state: _TestForeignCallIntrospectionBranchesState) -> None:
    """Execute part 3 of test_foreign_call_introspection_branches.

    Args:
        _state: Shared test state dictionary.
    """
    fc = _state["fc"]

    def non_int_tuple() -> tuple[str, str]:
        """Mock.

        Returns:
            tuple[str, str]: Computed output value from `non_int_tuple`.
        """
        return ("a", "b")

    assert adapt_backend_module(fc).require_symbol("infer_shape")(non_int_tuple) == ()

    # Return string AST
    def ast_str_return() -> BackendArg:
        """Mock.

        Returns:
            BackendArg: Computed output value from `ast_str_return`.
        """
        return ("a", "b")

    assert adapt_backend_module(fc).require_symbol("infer_shape")(ast_str_return) == ()

    def ast_int_return() -> BackendArg:
        """Mock.

        Returns:
            BackendArg: Computed output value from `ast_int_return`.
        """
        return 1

    assert adapt_backend_module(fc).require_symbol("infer_shape")(ast_int_return) == ()

    class _BareDummy:
        """Empty dummy class without attributes."""

    # Fallthrough
    assert adapt_backend_module(fc).require_symbol("infer_shape")(_BareDummy()) == ()


def test_foreign_call_introspection_branches() -> None:
    """Test fallback branches in introspection."""
    _state: _TestForeignCallIntrospectionBranchesState = {}
    _test_foreign_call_introspection_branches_part_1(_state)
    _test_foreign_call_introspection_branches_part_2(_state)
    _test_foreign_call_introspection_branches_part_3(_state)
    assert len(_state) > 0


def test_foreign_call_introspection_kwargs_only() -> None:
    """Test kwargs without args."""
    fc = ForeignCall()
    assert fc.infer_shape(output_shape=(8, 8)) == (8, 8)
    assert fc.infer_shape(output_shape=9) == (9,)
    assert fc.infer_shape(other_kwarg=1) == ()


def test_foreign_call_output_shape_scalar() -> None:
    """Test output_shape scalar handling."""
    fc = ForeignCall()

    class WithScalarOutShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = 5

    assert fc.infer_shape(WithScalarOutShape()) == (5,)


def test_foreign_call_inspect_signature() -> None:
    """Test fallback to inspect.signature."""
    fc = ForeignCall()

    class MyCallable(MockTestStub):
        """Mock callable."""

        def __call__(self) -> tuple[int, int]:
            """Call.

            Returns:
                tuple[int, int]: Computed output value from `__call__`.
            """
            return (0, 0)

    MyCallable.__call__.__annotations__["return"] = tuple[int, int]

    # typing.get_type_hints doesn't work directly on objects usually without accessing __call__
    assert fc.infer_shape(MyCallable()) == (int, int)


def test_foreign_call_fallback_args() -> None:
    """Test fallback args branch."""
    fc = ForeignCall()

    class WithShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.shape = (3, 3)

    assert fc.infer_shape(None, WithShape()) == (3, 3)


def test_foreign_call_ast_tuple_mismatch() -> None:
    """Test AST tuple elements mismatch."""
    fc = ForeignCall()

    def my_func() -> BackendArg:
        """Mock.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return (1, "not int")

    assert fc.infer_shape(my_func) == ()


def test_foreign_call_fallback_exceptions() -> None:
    """Test fallback exceptions in AST."""
    fc = ForeignCall()

    def bad_ast_func() -> None:
        """Mock."""
        pass

    import inspect

    # We monkeypatch inspect.getsource to throw OSError
    def mock_getsource(obj: BackendArg) -> None:
        """Test function mock_getsource.

        Args:
            obj: Test parameter `obj`.

        Raises:
            OSError: When raised during `mock_getsource` (Cannot get source).
        """
        raise OSError("Cannot get source")

    original_getsource = inspect.getsource
    pytest.MonkeyPatch().setattr(inspect, "getsource", mock_getsource, raising=False)
    try:
        assert fc.infer_shape(bad_ast_func) == ()
    finally:
        inspect.getsource = original_getsource


def test_foreign_call_fallback_empty() -> None:
    """Test fallback empty branch."""
    fc = ForeignCall()
    assert fc.infer_shape() == ()


def test_foreign_call_out_shape_tuple_branch() -> None:
    """Test branch where output_shape is tuple."""
    fc = ForeignCall()

    class WithTupleOutShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = (5, 5)

    assert fc.infer_shape(WithTupleOutShape()) == (5, 5)


def test_foreign_call_out_shape_kwarg_tuple_branch() -> None:
    """Test branch where output_shape kwarg is tuple."""
    fc = ForeignCall()
    assert fc.infer_shape(output_shape=(7, 7)) == (7, 7)


def test_foreign_call_fallback_func_with_no_returns() -> None:
    """Test fallback when func has no hints."""
    fc = ForeignCall()

    def my_func() -> None:
        """Mock."""
        pass

    assert fc.infer_shape(my_func) == ()


def test_foreign_call_exception_branch_inspect_signature() -> None:
    """Test exception branch for inspect_signature."""
    fc = ForeignCall()

    import inspect

    original_sig = inspect.signature

    def failing_sig(obj: BackendArg) -> None:
        """Mock.

        Args:
            obj: Test parameter `obj`.

        Raises:
            ValueError: When raised during `failing_sig` (Cannot get sig).
        """
        raise ValueError("Cannot get sig")

    pytest.MonkeyPatch().setattr(inspect, "signature", failing_sig, raising=False)
    try:

        def another_func() -> None:
            """Mock."""
            pass

        from ml_switcheroo_compiler.core.errors import ShapeInferenceError

        with pytest.raises(ShapeInferenceError, match="Cannot get sig"):
            fc.infer_shape(another_func)
    finally:
        inspect.signature = original_sig


def test_foreign_call_out_shapes_empty_tuple_branch() -> None:
    """Test branch where output_shapes kwarg is empty tuple."""
    fc = ForeignCall()

    class WithEmptyOutputShapes:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = ()

    assert fc.infer_shape(WithEmptyOutputShapes()) == ()


def test_foreign_call_out_shape_tuple_branch_2() -> None:
    """Test branch where output_shape is tuple (redundant)."""
    fc = ForeignCall()

    class WithTupleOutShape2:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = [5, 5]

    assert fc.infer_shape(WithTupleOutShape2()) == (5, 5)


def test_foreign_call_out_shapes_empty_tuple_branch_2() -> None:
    """Test branch where output_shapes kwarg is empty tuple (redundant)."""
    fc = ForeignCall()

    class WithEmptyOutputShapes2:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes: list[list[int]] = [[]]

    assert fc.infer_shape(WithEmptyOutputShapes2()) == ()

    class WithEmptyOutputShapes3:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = ()

    assert fc.infer_shape(WithEmptyOutputShapes3()) == ()


def test_foreign_call_out_nodes_metadata_branch() -> None:
    """Test branch where out_nodes exist but missing shape_metadata."""
    fc = ForeignCall()

    class MockSubNodeNoMetadata:
        """Mock."""

        pass

    class MockSubgraphNoMetadata:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.outputs = ["out1"]
            self.nodes = {"out1": MockSubNodeNoMetadata()}

    class WithSubgraphNoMetadata(MockTestStub):
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.subgraph = MockSubgraphNoMetadata()

    assert fc.infer_shape(WithSubgraphNoMetadata()) == ()
