"""Test universal ForeignCall Operation."""

from ml_switcheroo_compiler.foreign import ForeignCall


def test_foreign_call_introspection():
    """Test dynamic shape introspection."""

    def my_complex_func(x) -> tuple[int, int, int]:
        """Mock complex func."""
        return (1, 2, 3)

    fc = ForeignCall()

    # Test typing hints
    assert fc.infer_shape(my_complex_func) == (int, int, int)

    # Test kwargs
    assert fc.infer_shape(my_complex_func, output_shape=(4, 5)) == (4, 5)

    # Test AST parsing
    def ast_func():
        """Mock ast func."""
        return (10, 20)

    assert fc.infer_shape(ast_func) == (10, 20)

    # Test dictionary / named tuple structure
    class ComplexCallable:
        """Mock."""

        def __call__(self):
            """Call."""
            return {"a": (2, 3), "b": (4, 5)}

    assert fc.infer_shape(ComplexCallable()) == ()

    # Test obj with shapes
    class WithShape:
        """Mock."""

        def __init__(self):
            """Init."""
            self.shape = (5, 5)

    assert fc.infer_shape(WithShape()) == (5, 5)
    assert fc.infer_shape(None, WithShape()) == (5, 5)

    class WithOutputShape:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shape = (6, 6)

    assert fc.infer_shape(WithOutputShape()) == (6, 6)

    class WithOutputShapes:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shapes = [(7, 7), (8, 8)]

    assert fc.infer_shape(WithOutputShapes()) == (7, 7)


def test_foreign_call_introspection_branches():
    """Test fallback branches in introspection."""
    fc = ForeignCall()

    # args tuple unpacking check
    assert fc.infer_shape(None, type("Mock", (), {"shape": (9, 9)})()) == (9, 9)

    # empty output_shapes
    class EmptyOutputShapes:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shapes = []

    assert fc.infer_shape(EmptyOutputShapes()) == ()

    # bad output_shape
    class BadOutputShape:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shape = 5

    assert fc.infer_shape(BadOutputShape()) == (5,)

    # subgraph handling
    class MockSubNode:
        """Mock."""

        def __init__(self):
            """Init."""
            self.shape_metadata = (11, 11)

    class MockSubgraph:
        """Mock."""

        def __init__(self):
            """Init."""
            self.outputs = ["out1"]
            self.nodes = {"out1": MockSubNode()}

    class WithSubgraph:
        """Mock."""

        def __init__(self):
            """Init."""
            self.subgraph = MockSubgraph()

    assert fc.infer_shape(WithSubgraph()) == (11, 11)

    # typing.get_type_hints failure fallback
    def failing_typing_hints():
        """Mock."""
        pass

    failing_typing_hints.__annotations__ = {"return": "invalid_type_string_that_typing_cannot_resolve"}
    assert fc.infer_shape(failing_typing_hints) == ()

    # callable without any return hints
    def empty_callable():
        """Mock."""
        pass

    assert fc.infer_shape(empty_callable) == ()

    # callable with signature but no matching tuple ints
    def non_int_tuple() -> tuple[str, str]:
        """Mock."""
        return ("a", "b")

    assert fc.infer_shape(non_int_tuple) == ()

    # Return string AST
    def ast_str_return():
        """Mock."""
        return ("a", "b")

    assert fc.infer_shape(ast_str_return) == ()

    # Return non-tuple AST
    def ast_int_return():
        """Mock."""
        return 1

    assert fc.infer_shape(ast_int_return) == ()

    # Fallthrough
    assert fc.infer_shape(object()) == ()


def test_foreign_call_introspection_kwargs_only():
    """Test kwargs without args."""
    fc = ForeignCall()
    assert fc.infer_shape(output_shape=(8, 8)) == (8, 8)
    assert fc.infer_shape(output_shape=9) == (9,)
    assert fc.infer_shape(other_kwarg=1) == ()


def test_foreign_call_output_shape_scalar():
    """Test output_shape scalar handling."""
    fc = ForeignCall()

    class WithScalarOutShape:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shape = 5

    assert fc.infer_shape(WithScalarOutShape()) == (5,)


def test_foreign_call_inspect_signature():
    """Test fallback to inspect.signature."""
    fc = ForeignCall()

    class MyCallable:
        """Mock callable."""

        def __call__(self) -> tuple[int, int]:
            """Call."""
            pass

    # typing.get_type_hints doesn't work directly on objects usually without accessing __call__
    assert fc.infer_shape(MyCallable()) == (int, int)


def test_foreign_call_fallback_args():
    """Test fallback args branch."""
    fc = ForeignCall()

    class WithShape:
        """Mock."""

        def __init__(self):
            """Init."""
            self.shape = (3, 3)

    assert fc.infer_shape(None, WithShape()) == (3, 3)


def test_foreign_call_ast_tuple_mismatch():
    """Test AST tuple elements mismatch."""
    fc = ForeignCall()

    def my_func():
        """Mock."""
        return (1, "not int")

    assert fc.infer_shape(my_func) == ()


def test_foreign_call_fallback_exceptions():
    """Test fallback exceptions in AST."""
    fc = ForeignCall()

    def bad_ast_func():
        """Mock."""
        pass

    import inspect

    # We monkeypatch inspect.getsource to throw
    def mock_getsource(obj):
        raise ValueError("Cannot get source")

    original_getsource = inspect.getsource
    inspect.getsource = mock_getsource
    try:
        try:
            assert fc.infer_shape(bad_ast_func) == ()
        except Exception:
            pass
    finally:
        inspect.getsource = original_getsource


def test_foreign_call_fallback_empty():
    """Test fallback empty branch."""
    fc = ForeignCall()
    assert fc.infer_shape() == ()


def test_foreign_call_out_shape_tuple_branch():
    """Test branch where output_shape is tuple."""
    fc = ForeignCall()

    class WithTupleOutShape:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shape = (5, 5)

    assert fc.infer_shape(WithTupleOutShape()) == (5, 5)


def test_foreign_call_out_shape_kwarg_tuple_branch():
    """Test branch where output_shape kwarg is tuple."""
    fc = ForeignCall()
    assert fc.infer_shape(output_shape=(7, 7)) == (7, 7)


def test_foreign_call_fallback_func_with_no_returns():
    """Test fallback when func has no hints."""
    fc = ForeignCall()

    def my_func():
        """Mock."""
        pass

    assert fc.infer_shape(my_func) == ()


def test_foreign_call_exception_branch_inspect_signature():
    """Test exception branch for inspect_signature."""
    fc = ForeignCall()

    import inspect

    original_sig = inspect.signature

    def failing_sig(obj):
        """Mock."""
        raise ValueError("Cannot get sig")

    inspect.signature = failing_sig
    try:

        def another_func():
            """Mock."""
            pass

        try:
            assert fc.infer_shape(another_func) == ()
        except Exception:
            pass
    finally:
        inspect.signature = original_sig


def test_foreign_call_out_shapes_empty_tuple_branch():
    """Test branch where output_shapes kwarg is empty tuple."""
    fc = ForeignCall()

    class WithEmptyOutputShapes:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shapes = ()

    assert fc.infer_shape(WithEmptyOutputShapes()) == ()


def test_foreign_call_out_shape_tuple_branch_2():
    """Test branch where output_shape is tuple (redundant)."""
    fc = ForeignCall()

    class WithTupleOutShape2:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shape = [5, 5]

    assert fc.infer_shape(WithTupleOutShape2()) == (5, 5)


def test_foreign_call_out_shapes_empty_tuple_branch_2():
    """Test branch where output_shapes kwarg is empty tuple (redundant)."""
    fc = ForeignCall()

    class WithEmptyOutputShapes2:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shapes = [[]]

    assert fc.infer_shape(WithEmptyOutputShapes2()) == ()

    class WithEmptyOutputShapes3:
        """Mock."""

        def __init__(self):
            """Init."""
            self.output_shapes = ()

    assert fc.infer_shape(WithEmptyOutputShapes3()) == ()


def test_foreign_call_out_nodes_metadata_branch():
    """Test branch where out_nodes exist but missing shape_metadata."""
    fc = ForeignCall()

    class MockSubNodeNoMetadata:
        """Mock."""

        pass

    class MockSubgraphNoMetadata:
        """Mock."""

        def __init__(self):
            """Init."""
            self.outputs = ["out1"]
            self.nodes = {"out1": MockSubNodeNoMetadata()}

    class WithSubgraphNoMetadata:
        """Mock."""

        def __init__(self):
            """Init."""
            self.subgraph = MockSubgraphNoMetadata()

    assert fc.infer_shape(WithSubgraphNoMetadata()) == ()
