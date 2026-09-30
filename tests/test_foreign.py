# ruff: noqa
from ml_switcheroo_compiler.ir.core import LogicalGraph
from ml_switcheroo_compiler.transforms.foreign import (
    _extract_jaxpr_constants,
    _handle_fx_call_function,
    _handle_fx_output,
    _handle_fx_placeholder,
    _translate_jax_equation,
    ingest_jaxpr,
    ingest_torch_fx,
)
from ml_switcheroo_compiler.foreign import ForeignCall
import pytest

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


class MockFXNode:
    """Mock Torch FX node."""

    def __init__(self, name: str, op: str, target: object = None, args: tuple = ()) -> None:
        self.name = name
        self.op = op
        self.target = target
        self.args = args

    def __str__(self) -> str:
        return self.name


class MockFXGraph:
    """Mock Torch FX graph."""

    def __init__(self, nodes: list) -> None:
        self.nodes = nodes


class MockFXGraphModule:
    """Mock Torch FX GraphModule."""

    def __init__(self, nodes: list = None, has_graph: bool = True) -> None:
        if has_graph:
            self.graph = MockFXGraph(nodes or [])


def test_ingest_torch_fx() -> None:
    """Test ingest_torch_fx under various graph configurations."""
    with pytest.raises(ValueError, match="Torch FX GraphModule cannot be None"):
        ingest_torch_fx(None)

    # Missing graph attribute
    empty_res = ingest_torch_fx(object())
    assert len(empty_res.nodes) == 0

    # Missing nodes attribute in graph
    missing_nodes_res = ingest_torch_fx(type("BadMod", (), {"graph": object()})())
    assert len(missing_nodes_res.nodes) == 0

    # Create dummy function targets
    class TargetAdd:
        __name__ = "torch.add"

    class TargetMul:
        __name__ = "torch.mul"

    class TargetSub:
        __name__ = "torch.sub"

    node_x = MockFXNode("x", "placeholder")
    node_y = MockFXNode("y", "placeholder")
    node_add = MockFXNode("add_1", "call_function", target=TargetAdd(), args=(node_x, node_y))
    node_mul = MockFXNode("mul_1", "call_function", target=TargetMul(), args=(node_add, 2.0))
    node_unknown = MockFXNode("un_1", "call_function", target="custom_sub", args=(MockFXNode("untracked", "placeholder"),))
    node_ignored = MockFXNode("ignore_me", "get_attr")
    node_out_tuple = MockFXNode("out", "output", args=((node_mul, 999),))

    gm = MockFXGraphModule([node_x, node_y, node_add, node_mul, node_unknown, node_ignored, node_out_tuple])
    res = ingest_torch_fx(gm)

    assert "add_1" in res.nodes
    assert res.nodes["add_1"].op_type == "Add"
    assert res.nodes["add_1"].inputs == ["x", "y"]

    assert "mul_1" in res.nodes
    assert res.nodes["mul_1"].op_type == "Mul"
    assert res.nodes["mul_1"].inputs == ["add_1", "2.0"]

    assert "un_1" in res.nodes
    assert res.nodes["un_1"].op_type == "Unknown"
    assert res.nodes["un_1"].inputs == ["untracked"]

    assert res.outputs == ["mul_1"]

    # Test output node with single named arg
    gm_single = MockFXGraphModule([node_x, MockFXNode("out", "output", args=(node_x,))])
    res_single = ingest_torch_fx(gm_single)
    assert res_single.outputs == ["x"]

    # Test output node with non-named single arg
    gm_raw = MockFXGraphModule([MockFXNode("out", "output", args=(123,))])
    res_raw = ingest_torch_fx(gm_raw)
    assert res_raw.outputs == []


def test_ingest_jaxpr() -> None:
    """Test ingest_jaxpr and related translation utilities."""
    with pytest.raises(ValueError, match="JAX jaxpr cannot be None"):
        ingest_jaxpr(None)

    # Empty jaxpr without eqns or consts
    res_empty = ingest_jaxpr(object())
    assert len(res_empty.nodes) == 0

    class MockConstWithShape:
        shape = (4, 5)

    const_var = object()
    const_val = MockConstWithShape()
    jaxpr_obj = type(
        "MockJaxprFull",
        (),
        {
            "consts": [const_val],
            "constvars": [const_var],
        },
    )()

    res_const = LogicalGraph()
    _extract_jaxpr_constants(jaxpr_obj, res_const)
    assert str(id(const_var)) in res_const.nodes
    assert res_const.nodes[str(id(const_var))].shape_metadata == (4, 5)

    # Test equation translation: add, mul, unknown, and no outvars
    class PrimAdd:
        name = "add"

    class PrimMul:
        name = "mul"

    class DummyVar:
        pass

    in1 = DummyVar()
    in2 = DummyVar()
    out1 = DummyVar()

    eqn_add = type("Eqn", (), {"primitive": PrimAdd(), "invars": [in1, in2], "outvars": [out1]})()
    eqn_mul = type("Eqn", (), {"primitive": PrimMul(), "invars": [out1], "outvars": [DummyVar()]})()
    eqn_other = type("Eqn", (), {"primitive": "cos", "invars": [], "outvars": []})()

    full_jaxpr = type(
        "FullJaxpr",
        (),
        {
            "consts": [],
            "eqns": [eqn_add, eqn_mul, eqn_other],
        },
    )()

    res_graph = ingest_jaxpr(full_jaxpr)
    assert str(id(out1)) in res_graph.nodes
    assert res_graph.nodes[str(id(out1))].op_type == "Add"
    assert "out" in res_graph.nodes
    assert res_graph.nodes["out"].op_type == "Unknown"


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
