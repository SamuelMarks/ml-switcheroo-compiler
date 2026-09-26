# ruff: noqa
from ml_switcheroo_compiler.ir.core import LogicalGraph
from ml_switcheroo_compiler.transforms.foreign import _extract_jaxpr_constants
from ml_switcheroo_compiler.foreign import ForeignCall

"Tests for foreign transform logic."


class MockConst:
    """Mock constant."""


class MockJaxpr:
    """Mock JAX program."""

    def __init__(self, consts: list = None) -> None:
        """Initialize.

        Args:
            consts: Optional constants list.
        """
        self.consts = consts or []
        self.eqns = []


def test_extract_jaxpr_constants() -> None:
    """Test jaxpr constant extraction."""
    graph = LogicalGraph()
    jaxpr_empty = MockJaxpr()
    _extract_jaxpr_constants(jaxpr_empty, graph)
    jaxpr_with = MockJaxpr([MockConst()])
    _extract_jaxpr_constants(jaxpr_with, graph)


def test_extract_jaxpr_constants_nodes_added() -> None:
    """Test that constants are correctly added to the graph."""
    graph = LogicalGraph()
    jaxpr_with = MockJaxpr([MockConst()])
    _extract_jaxpr_constants(jaxpr_with, graph)
    assert "const_0" in graph.nodes
    assert graph.nodes["const_0"].op_type == "Constant"


def test_generic_utils_stubs():
    """Test generic utils stubs."""
    from ml_switcheroo_compiler.utils.generic_utils import (
        CustomObjectScope,
        clear_session,
        deserialize_keras_object,
        disable_interactive_logging,
        enable_interactive_logging,
        get_custom_objects,
        get_registered_name,
        get_registered_object,
        is_interactive_logging_enabled,
        is_keras_tensor,
        register_keras_serializable,
        serialize_keras_object,
        standardize_dtype,
    )

    clear_session()
    with CustomObjectScope():
        pass
    assert deserialize_keras_object() is None
    enable_interactive_logging()
    assert is_interactive_logging_enabled() is True
    disable_interactive_logging()
    assert is_interactive_logging_enabled() is False
    assert get_custom_objects() == {}
    assert get_registered_name() == ""
    assert get_registered_object() is None
    assert is_keras_tensor() is False

    @register_keras_serializable()
    class Dummy:
        pass

    assert serialize_keras_object() is None
    assert standardize_dtype() is None


def _sample_ast_tuple(x: object) -> object:
    """Sample AST function returning int tuple."""
    return (11, 22, 33)


def _sample_ast_int(x: object) -> object:
    """Sample AST function returning int."""
    return 42


def _sample_ast_vars(a: object, b: object) -> object:
    """Sample AST function returning variable tuple."""
    return (a, b)


def _sample_ast_none() -> None:
    """Sample AST function returning None."""
    return


def test_foreign_coverage():
    """Test all branches of ForeignCall.infer_shape."""
    import inspect
    from typing import Tuple

    op = ForeignCall()

    # 1. No arguments and None output_shape
    assert op.infer_shape() == ()
    assert op.infer_shape(output_shape=None) == ()

    # 2. Kwarg output_shape as list/tuple and as non-sequence
    assert op.infer_shape(output_shape=[2, 3]) == (2, 3)
    assert op.infer_shape(output_shape=(4, 5)) == (4, 5)
    assert op.infer_shape(output_shape=7) == (7,)

    # 3. First arg has shape
    class DummyTensor:
        shape = (1, 2)

    assert op.infer_shape(DummyTensor()) == (1, 2)

    # 4. First arg has output_shape
    class DummyOutputShapeTuple:
        output_shape = (3, 4)

    class DummyOutputShapeScalar:
        output_shape = 9

    assert op.infer_shape(DummyOutputShapeTuple()) == (3, 4)
    assert op.infer_shape(DummyOutputShapeScalar()) == (9,)

    # 5. First arg has output_shapes
    class DummyOutputShapesList:
        output_shapes = [(5, 6), (7, 8)]

    class DummyOutputShapesEmpty:
        output_shapes = []

    class DummyOutputShapesNonSeq:
        output_shapes = 123

    assert op.infer_shape(DummyOutputShapesList()) == (5, 6)
    assert op.infer_shape(DummyOutputShapesEmpty()) == ()
    assert op.infer_shape(DummyOutputShapesNonSeq()) == ()

    # 6. First arg has subgraph with outputs
    class NodeWithMeta:
        shape_metadata = (10, 20)

    class SubgraphValid:
        outputs = ["out1"]
        nodes = {"out1": NodeWithMeta()}

    class SubgraphMissingOut:
        outputs = ["out_missing"]
        nodes = {}

    class NodeNoMeta:
        shape_metadata = None

    class SubgraphNoMeta:
        outputs = ["out2"]
        nodes = {"out2": NodeNoMeta()}

    class NodeEmptyMeta:
        shape_metadata = ()

    class SubgraphEmptyMeta:
        outputs = ["out3"]
        nodes = {"out3": NodeEmptyMeta()}

    class Container:
        def __init__(self, sg):
            self.subgraph = sg

    assert op.infer_shape(Container(SubgraphValid())) == (10, 20)
    assert op.infer_shape(Container(SubgraphMissingOut())) == ()
    assert op.infer_shape(Container(SubgraphNoMeta())) == ()
    assert op.infer_shape(Container(SubgraphEmptyMeta())) == ()

    # 7. Callable first argument
    # 7a. typing.get_type_hints with return annotation
    def fn_hints(x: int) -> Tuple[int, int]:
        return (x, x)

    assert op.infer_shape(fn_hints) == (int, int)

    # 7b. hints with no "return"
    def fn_no_ret(x: int):
        pass

    assert op.infer_shape(fn_no_ret) == ()

    # 7c. hints with return not matching all(int or type)
    def fn_str_ret() -> Tuple[str, ...]:
        return ("a",)

    assert op.infer_shape(fn_str_ret) == ()

    # 7d. Callable that fails get_type_hints but signature succeeds
    class CallableWithSignature:
        def __init__(self, ret_anno):
            self.__annotations__ = {"return": "NonExistentTypeReference"}
            self._ret_anno = ret_anno

        def __call__(self, *args):
            pass

    mock_callable = CallableWithSignature(Tuple[int, int, int])
    mock_callable.__signature__ = inspect.Signature(return_annotation=Tuple[int, int, int])
    assert op.infer_shape(mock_callable) == (int, int, int)

    # 7e. Callable with signature raising Exception
    class BrokenSignatureCallable:
        def __call__(self):
            pass

        @property
        def __signature__(self):
            raise RuntimeError("broken sig")

    assert op.infer_shape(BrokenSignatureCallable()) == ()

    # 7f. AST parsing: function returning constant int tuple without type hints
    assert op.infer_shape(_sample_ast_tuple) == (11, 22, 33)

    # AST parsing: function returning non-tuple (e.g. constant int)
    assert op.infer_shape(_sample_ast_int) == ()

    # AST parsing: function returning tuple with non-constants
    assert op.infer_shape(_sample_ast_vars) == ()

    # AST parsing: function returning empty / None
    assert op.infer_shape(_sample_ast_none) == ()

    # AST parsing: callable where inspect.getsource fails (e.g. builtin)
    assert op.infer_shape(len) == ()

    # 8. len(args) > 1 and args[1] has shape
    class PlainObj:
        pass

    assert op.infer_shape(PlainObj(), DummyTensor()) == (1, 2)
    assert op.infer_shape(PlainObj(), PlainObj()) == ()
