# ruff: noqa: C901, PLR0912, PLR0915, E402, F401, F811, F841, E712, D301, UP006, UP035
"""Test universal ForeignCall Operation."""

from collections.abc import Callable
from typing import Tuple, cast

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.foreign import ForeignCall


def test_foreign_call_introspection() -> None:
    """Test dynamic shape introspection."""

    def my_complex_func(x: EagerValue) -> Tuple[int, int, int]:
        """Mock complex func.

        Args:
            x: Test parameter `x`.
        """
        return (1, 2, 3)

    fc = ForeignCall()

    # Test typing hints
    assert cast("EagerTensorProtocol", fc.infer_shape(my_complex_func)) == cast("EagerTensorProtocol", (int, int, int))

    # Test kwargs
    assert fc.infer_shape(my_complex_func, output_shape=(4, 5)) == (4, 5)

    # Test AST parsing
    def ast_func() -> EagerValue:
        """Mock ast func."""
        return (10, 20)

    assert fc.infer_shape(ast_func) == (10, 20)

    # Test dictionary / named tuple structure
    class ComplexCallable:
        """Mock."""

        def __call__(self) -> EagerValue:
            """Call."""
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

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithOutputShape)()) == (6, 6)

    class WithOutputShapes:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = [(7, 7), (8, 8)]

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithOutputShapes)()) == (7, 7)


def _test_foreign_call_introspection_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_foreign_call_introspection_branches.

    Args:
        _state: Shared test state dictionary.
    """
    fc = ForeignCall()

    # args tuple unpacking check
    assert fc.infer_shape(None, type("Mock", (), {"shape": (9, 9)})()) == (9, 9)

    # empty output_shapes
    class EmptyOutputShapes:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = cast("list[EagerTensorProtocol]", [])

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", EmptyOutputShapes)()) == ()

    # bad output_shape
    class BadOutputShape:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = 5

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", BadOutputShape)()) == (5,)

    # subgraph handling
    class MockSubNode:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.shape_metadata = (11, 11)

    getattr(_state, "update")({"BadOutputShape": BadOutputShape, "EmptyOutputShapes": EmptyOutputShapes, "MockSubNode": MockSubNode, "fc": fc})
    _state.update(locals())


def _test_foreign_call_introspection_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
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

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithSubgraph)()) == (11, 11)

    # typing.get_type_hints failure fallback
    def failing_typing_hints() -> None:
        """Mock."""
        pass

    failing_typing_hints.__annotations__ = {"return": "invalid_type_string_that_typing_cannot_resolve"}
    assert fc.infer_shape(failing_typing_hints) == ()

    # callable without any return hints
    def empty_callable() -> None:
        """Mock."""
        pass

    assert fc.infer_shape(empty_callable) == ()
    getattr(_state, "update")({"MockSubgraph": MockSubgraph, "WithSubgraph": WithSubgraph, "empty_callable": empty_callable, "failing_typing_hints": failing_typing_hints})
    _state.update(locals())


def _test_foreign_call_introspection_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_foreign_call_introspection_branches.

    Args:
        _state: Shared test state dictionary.
    """
    fc = _state["fc"]

    def non_int_tuple() -> tuple[str, str]:
        """Mock."""
        return ("a", "b")

    assert fc.infer_shape(non_int_tuple) == ()

    # Return string AST
    def ast_str_return() -> EagerValue:
        """Mock."""
        return ("a", "b")

    assert fc.infer_shape(ast_str_return) == ()

    def ast_int_return() -> EagerValue:
        """Mock."""
        return 1

    assert fc.infer_shape(ast_int_return) == ()

    # Fallthrough
    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", object)()) == ()
    getattr(_state, "update")({"ast_int_return": ast_int_return, "ast_str_return": ast_str_return, "non_int_tuple": non_int_tuple})


def test_foreign_call_introspection_branches() -> None:
    """Test fallback branches in introspection."""
    _state: dict[str, EagerTensorProtocol] = {}
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

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithScalarOutShape)()) == (5,)


def test_foreign_call_inspect_signature() -> None:
    """Test fallback to inspect.signature."""
    fc = ForeignCall()

    class MyCallable:
        """Mock callable."""

        def __call__(self) -> Tuple[int, int]:
            """Call."""
            return (0, 0)

    # typing.get_type_hints doesn't work directly on objects usually without accessing __call__
    assert cast("EagerTensorProtocol", fc.infer_shape(MyCallable())) == cast("EagerTensorProtocol", (int, int))


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

    def my_func() -> EagerValue:
        """Mock."""
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
    def mock_getsource(obj: EagerValue) -> None:
        """Test function mock_getsource.

        Args:
            obj: Test parameter `obj`.
        """
        raise OSError("Cannot get source")

    original_getsource = inspect.getsource
    setattr(inspect, "getsource", mock_getsource)
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

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithTupleOutShape)()) == (5, 5)


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

    def failing_sig(obj: EagerValue) -> None:
        """Mock.

        Args:
            obj: Test parameter `obj`.
        """
        raise ValueError("Cannot get sig")

    setattr(inspect, "signature", failing_sig)
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

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithEmptyOutputShapes)()) == ()


def test_foreign_call_out_shape_tuple_branch_2() -> None:
    """Test branch where output_shape is tuple (redundant)."""
    fc = ForeignCall()

    class WithTupleOutShape2:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shape = [5, 5]

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithTupleOutShape2)()) == (5, 5)


def test_foreign_call_out_shapes_empty_tuple_branch_2() -> None:
    """Test branch where output_shapes kwarg is empty tuple (redundant)."""
    fc = ForeignCall()

    class WithEmptyOutputShapes2:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = cast("EagerTensorProtocol", [[]])

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithEmptyOutputShapes2)()) == ()

    class WithEmptyOutputShapes3:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.output_shapes = ()

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithEmptyOutputShapes3)()) == ()


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

    class WithSubgraphNoMetadata:
        """Mock."""

        def __init__(self) -> None:
            """Init."""
            self.subgraph = MockSubgraphNoMetadata()

    assert cast("EagerCallableProtocol", fc.infer_shape)(cast("EagerCallableProtocol", WithSubgraphNoMetadata)()) == ()
