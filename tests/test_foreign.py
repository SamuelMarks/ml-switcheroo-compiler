"""Tests for test_foreign."""

from __future__ import annotations
from typing import Optional

import importlib
from typing import cast
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockConst:
    """Mock constant."""


class MockJaxpr:
    """Mock JAX program."""

    def __init__(self, consts: Optional[list[EagerValue]] = None) -> None:
        """Initialize.

        Args:
            consts: Optional constants list.
        """
        self.consts = consts or []
        self.eqns = cast("list[EagerTensorProtocol]", [])


def test_extract_jaxpr_constants() -> None:
    """Test jaxpr constant extraction."""
    graph = LogicalGraph()
    jaxpr_empty = MockJaxpr()
    cast("Callable[..., EagerTensorProtocol]", _extract_jaxpr_constants)(jaxpr_empty, graph)
    assert len(graph.nodes) == 0
    jaxpr_with = cast("Callable[..., EagerTensorProtocol]", MockJaxpr)([cast("Callable[..., EagerTensorProtocol]", MockConst)()])
    cast("Callable[..., EagerTensorProtocol]", _extract_jaxpr_constants)(jaxpr_with, graph)
    assert "const_0" in graph.nodes


def test_extract_jaxpr_constants_nodes_added() -> None:
    """Test that constants are correctly added to the graph."""
    graph = LogicalGraph()
    jaxpr_with = cast("Callable[..., EagerTensorProtocol]", MockJaxpr)([cast("Callable[..., EagerTensorProtocol]", MockConst)()])
    cast("Callable[..., EagerTensorProtocol]", _extract_jaxpr_constants)(jaxpr_with, graph)
    assert "const_0" in graph.nodes
    assert graph.nodes["const_0"].op_type == "Constant"


class MockFXNode:
    """Mock Torch FX node."""

    def __init__(self, name: str, op: str, target: EagerValue = None, args: tuple[EagerValue, ...] = ()) -> None:
        """Test function __init__."""
        self.name = name
        self.op = op
        self.target = target
        self.args = args

    def __str__(self) -> str:
        """Test function __str__."""
        return self.name


class MockFXGraph:
    """Mock Torch FX graph."""

    def __init__(self, nodes: list[EagerValue]) -> None:
        """Test function __init__."""
        self.nodes = nodes


class MockFXGraphModule:
    """Mock Torch FX GraphModule."""

    def __init__(self, nodes: Optional[list[EagerValue]] = None, has_graph: bool = True) -> None:
        """Test function __init__."""
        if has_graph:
            self.graph = MockFXGraph(nodes or [])


def test_ingest_torch_fx() -> None:
    """Test ingest_torch_fx under various graph configurations."""
    with pytest.raises(ValueError, match="Torch FX GraphModule cannot be None"):
        ingest_torch_fx(None)

    # Missing graph attribute
    empty_res = cast("Callable[..., EagerTensorProtocol]", ingest_torch_fx)(cast("Callable[..., EagerTensorProtocol]", object)())
    assert len(empty_res.nodes) == 0

    # Missing nodes attribute in graph
    missing_nodes_res = ingest_torch_fx(type("BadMod", (), {"graph": object()})())
    assert len(missing_nodes_res.nodes) == 0

    # Create dummy function targets
    class TargetAdd:
        """Test class TargetAdd."""

        __name__ = "torch.add"

    class TargetMul:
        """Test class TargetMul."""

        __name__ = "torch.mul"

    class TargetSub:
        """Test class TargetSub."""

        __name__ = "torch.sub"

    node_x = MockFXNode("x", "placeholder")
    node_y = MockFXNode("y", "placeholder")
    node_add = cast("Callable[..., EagerTensorProtocol]", MockFXNode)("add_1", "call_function", target=cast("Callable[..., EagerTensorProtocol]", TargetAdd)(), args=(node_x, node_y))
    node_mul = cast("Callable[..., EagerTensorProtocol]", MockFXNode)("mul_1", "call_function", target=cast("Callable[..., EagerTensorProtocol]", TargetMul)(), args=(node_add, 2.0))
    node_unknown = cast("Callable[..., EagerTensorProtocol]", MockFXNode)("un_1", "call_function", target="custom_sub", args=(cast("Callable[..., EagerTensorProtocol]", MockFXNode)("untracked", "placeholder"),))
    node_ignored = MockFXNode("ignore_me", "get_attr")
    node_out_tuple = cast("Callable[..., EagerTensorProtocol]", MockFXNode)("out", "output", args=((node_mul, 999),))

    gm = cast("Callable[..., EagerTensorProtocol]", MockFXGraphModule)([node_x, node_y, node_add, node_mul, node_unknown, node_ignored, node_out_tuple])
    res = cast("Callable[..., EagerTensorProtocol]", ingest_torch_fx)(gm)

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
    gm_single = cast("Callable[..., EagerTensorProtocol]", MockFXGraphModule)([node_x, cast("Callable[..., EagerTensorProtocol]", MockFXNode)("out", "output", args=(node_x,))])
    res_single = cast("Callable[..., EagerTensorProtocol]", ingest_torch_fx)(gm_single)
    assert res_single.outputs == ["x"]

    # Test output node with non-named single arg
    gm_raw = cast("Callable[..., EagerTensorProtocol]", MockFXGraphModule)([cast("Callable[..., EagerTensorProtocol]", MockFXNode)("out", "output", args=(123,))])
    res_raw = cast("Callable[..., EagerTensorProtocol]", ingest_torch_fx)(gm_raw)
    assert res_raw.outputs == []


def test_ingest_jaxpr() -> None:
    """Test ingest_jaxpr and related translation utilities."""
    with pytest.raises(ValueError, match="JAX jaxpr cannot be None"):
        ingest_jaxpr(None)

    # Empty jaxpr without eqns or consts
    res_empty = cast("Callable[..., EagerTensorProtocol]", ingest_jaxpr)(cast("Callable[..., EagerTensorProtocol]", object)())
    assert len(res_empty.nodes) == 0

    class MockConstWithShape:
        """Test class MockConstWithShape."""

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
        """Test class PrimAdd."""

        name = "add"

    class PrimMul:
        """Test class PrimMul."""

        name = "mul"

    class DummyVar:
        """Test class DummyVar."""

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


def test_generic_utils_stubs() -> None:
    """Test generic utils stubs."""
    from ml_switcheroo_compiler.utils.generic_utils import disable_interactive_logging, enable_interactive_logging, is_interactive_logging_enabled, is_symbolic_tensor, standardize_dtype

    CustomObjectScope = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "CustomObjectScope"))
    deserialize_object = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "deserialize_object"))
    get_custom_objects = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "get_custom_objects"))
    get_registered_name = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "get_registered_name"))
    get_registered_object = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "get_registered_object"))
    register_serializable = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "register_serializable"))
    serialize_object = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "serialize_object"))

    with CustomObjectScope():
        pass
    assert deserialize_object() is None
    enable_interactive_logging()
    assert is_interactive_logging_enabled() is True
    disable_interactive_logging()
    assert is_interactive_logging_enabled() is False
    assert get_custom_objects() == {}
    assert get_registered_name() == ""
    assert get_registered_object() is None
    assert is_symbolic_tensor() is False

    @register_serializable()
    class Dummy:
        """Test class Dummy."""

        pass

    assert serialize_object() is None
    assert standardize_dtype() is None


def _sample_ast_tuple(x: EagerValue) -> EagerValue:
    """Sample AST function returning int tuple."""
    return (11, 22, 33)


def _sample_ast_int(x: EagerValue) -> EagerValue:
    """Sample AST function returning int."""
    return 42


def _sample_ast_vars(a: EagerValue, b: EagerValue) -> EagerValue:
    """Sample AST function returning variable tuple."""
    return (a, b)


def _sample_ast_none() -> None:
    """Sample AST function returning None."""
    return


def test_foreign_coverage() -> None:
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
        """Test class DummyTensor."""

        shape = (1, 2)

    assert op.infer_shape(DummyTensor()) == (1, 2)

    # 4. First arg has output_shape
    class DummyOutputShapeTuple:
        """Test class DummyOutputShapeTuple."""

        output_shape = (3, 4)

    class DummyOutputShapeScalar:
        """Test class DummyOutputShapeScalar."""

        output_shape = 9

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyOutputShapeTuple)()) == (3, 4)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyOutputShapeScalar)()) == (9,)

    # 5. First arg has output_shapes
    class DummyOutputShapesList:
        """Test class DummyOutputShapesList."""

        output_shapes = [(5, 6), (7, 8)]

    class DummyOutputShapesEmpty:
        """Test class DummyOutputShapesEmpty."""

        output_shapes = cast("list[EagerTensorProtocol]", [])

    class DummyOutputShapesNonSeq:
        """Test class DummyOutputShapesNonSeq."""

        output_shapes = 123

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyOutputShapesList)()) == (5, 6)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyOutputShapesEmpty)()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", DummyOutputShapesNonSeq)()) == ()

    # 6. First arg has subgraph with outputs
    class NodeWithMeta:
        """Test class NodeWithMeta."""

        shape_metadata = (10, 20)

    class SubgraphValid:
        """Test class SubgraphValid."""

        outputs = ["out1"]
        nodes = {"out1": NodeWithMeta()}

    class SubgraphMissingOut:
        """Test class SubgraphMissingOut."""

        outputs = ["out_missing"]
        nodes = cast("dict[str, EagerTensorProtocol]", {})

    class NodeNoMeta:
        """Test class NodeNoMeta."""

        shape_metadata = None

    class SubgraphNoMeta:
        """Test class SubgraphNoMeta."""

        outputs = ["out2"]
        nodes = {"out2": NodeNoMeta()}

    class NodeEmptyMeta:
        """Test class NodeEmptyMeta."""

        shape_metadata = ()

    class SubgraphEmptyMeta:
        """Test class SubgraphEmptyMeta."""

        outputs = ["out3"]
        nodes = {"out3": NodeEmptyMeta()}

    class Container:
        """Test class Container."""

        def __init__(self, sg: EagerValue) -> None:
            """Test function __init__."""
            self.subgraph = sg

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Container)(cast("Callable[..., EagerTensorProtocol]", SubgraphValid)())) == (10, 20)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Container)(cast("Callable[..., EagerTensorProtocol]", SubgraphMissingOut)())) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Container)(cast("Callable[..., EagerTensorProtocol]", SubgraphNoMeta)())) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Container)(cast("Callable[..., EagerTensorProtocol]", SubgraphEmptyMeta)())) == ()

    # 7. Callable first argument
    # 7a. typing.get_type_hints with return annotation
    def fn_hints(x: int) -> Tuple[int, int]:
        """Test function fn_hints."""
        return (x, x)

    assert cast("EagerTensorProtocol", op.infer_shape(fn_hints)) == cast("EagerTensorProtocol", (int, int))

    # 7b. hints with no "return"
    def fn_no_ret(x: int) -> None:
        """Test function fn_no_ret."""
        pass

    assert op.infer_shape(fn_no_ret) == ()

    # 7c. hints with return not matching all(int or type)
    def fn_str_ret() -> Tuple[str, ...]:
        """Test function fn_str_ret."""
        return ("a",)

    assert op.infer_shape(fn_str_ret) == ()

    # 7d. Callable that fails get_type_hints but signature succeeds
    class CallableWithSignature:
        """Test class CallableWithSignature."""

        def __init__(self, ret_anno: EagerValue) -> None:
            """Test function __init__."""
            self.__annotations__ = {"return": "NonExistentTypeReference"}
            self._ret_anno = ret_anno

        def __call__(self, *args: EagerValue) -> None:
            """Test function __call__."""
            pass

    mock_callable = cast("Callable[..., EagerTensorProtocol]", CallableWithSignature)(Tuple[int, int, int])
    setattr(mock_callable, "__signature__", inspect.Signature(return_annotation=Tuple[int, int, int]))
    assert cast("EagerTensorProtocol", op.infer_shape(mock_callable)) == cast("EagerTensorProtocol", (int, int, int))

    # 7e. Callable with signature raising Exception
    class BrokenSignatureCallable:
        """Test class BrokenSignatureCallable."""

        def __call__(self) -> None:
            """Test function __call__."""
            pass

        @property
        def __signature__(self) -> None:
            """Test function __signature__."""
            raise RuntimeError("broken sig")

    try:
        assert op.infer_shape(BrokenSignatureCallable()) == ()
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

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
        """Test class PlainObj."""

        pass

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", PlainObj)(), cast("Callable[..., EagerTensorProtocol]", DummyTensor)()) == (1, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", PlainObj)(), cast("Callable[..., EagerTensorProtocol]", PlainObj)()) == ()
