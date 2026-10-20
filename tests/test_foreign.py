"""Tests for test_foreign."""

from __future__ import annotations
from ml_switcheroo_compiler.backends.eager_registry import EagerValue

import importlib
from collections.abc import Callable
import typing

# ruff: noqa
from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
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


class MockConst:
    """Mock constant."""


class MockJaxpr:
    """Mock JAX program."""

    def __init__(self, consts: typing.Optional[list[EagerValue]] = None) -> None:
        """Initialize.

        Args:
            consts: Optional constants list.
        """
        self.consts = consts or []
        self.eqns = typing.cast("list[EagerTensorProtocol]", [])


def test_extract_jaxpr_constants() -> None:
    """Test jaxpr constant extraction."""
    graph = LogicalGraph()
    jaxpr_empty = MockJaxpr()
    typing.cast("EagerCallableProtocol", _extract_jaxpr_constants)(jaxpr_empty, graph)
    assert len(graph.nodes) == 0
    jaxpr_with = typing.cast("EagerCallableProtocol", MockJaxpr)([typing.cast("EagerCallableProtocol", MockConst)()])
    typing.cast("EagerCallableProtocol", _extract_jaxpr_constants)(jaxpr_with, graph)
    assert "const_0" in graph.nodes


def test_extract_jaxpr_constants_nodes_added() -> None:
    """Test that constants are correctly added to the graph."""
    graph = LogicalGraph()
    jaxpr_with = typing.cast("EagerCallableProtocol", MockJaxpr)([typing.cast("EagerCallableProtocol", MockConst)()])
    typing.cast("EagerCallableProtocol", _extract_jaxpr_constants)(jaxpr_with, graph)
    assert "const_0" in graph.nodes
    assert graph.nodes["const_0"].op_type == "Constant"


class MockFXNode:
    """Mock Torch FX node."""

    def __init__(self, name: str, op: str, target: EagerValue = None, args: tuple[EagerValue, ...] = ()) -> None:
        """Test function __init__.

        Args:
            name: Test parameter `name`.
            op: Test parameter `op`.
            target: Test parameter `target`.
            args: Test parameter `args`.
        """
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
        """Test function __init__.

        Args:
            nodes: Test parameter `nodes`.
        """
        self.nodes = nodes


class MockFXGraphModule:
    """Mock Torch FX GraphModule."""

    def __init__(self, nodes: typing.Optional[list[EagerValue]] = None, has_graph: bool = True) -> None:
        """Test function __init__.

        Args:
            nodes: Test parameter `nodes`.
            has_graph: Test parameter `has_graph`.
        """
        if has_graph:
            self.graph = MockFXGraph(nodes or [])


def test_ingest_torch_fx() -> None:
    """Test ingest_torch_fx under various graph configurations."""
    with pytest.raises(ValueError, match="Torch FX GraphModule cannot be None"):
        ingest_torch_fx(None)

    # Missing graph attribute
    empty_res = typing.cast("EagerCallableProtocol", ingest_torch_fx)(typing.cast("EagerCallableProtocol", object)())
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
    node_add = typing.cast("EagerCallableProtocol", MockFXNode)("add_1", "call_function", target=typing.cast("EagerCallableProtocol", TargetAdd)(), args=(node_x, node_y))
    node_mul = typing.cast("EagerCallableProtocol", MockFXNode)("mul_1", "call_function", target=typing.cast("EagerCallableProtocol", TargetMul)(), args=(node_add, 2.0))
    node_unknown = typing.cast("EagerCallableProtocol", MockFXNode)("un_1", "call_function", target="custom_sub", args=(typing.cast("EagerCallableProtocol", MockFXNode)("untracked", "placeholder"),))
    node_ignored = MockFXNode("ignore_me", "get_attr")
    node_out_tuple = typing.cast("EagerCallableProtocol", MockFXNode)("out", "output", args=((node_mul, 999),))

    gm = typing.cast("EagerCallableProtocol", MockFXGraphModule)([node_x, node_y, node_add, node_mul, node_unknown, node_ignored, node_out_tuple])
    res = typing.cast("EagerCallableProtocol", ingest_torch_fx)(gm)

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
    gm_single = typing.cast("EagerCallableProtocol", MockFXGraphModule)([node_x, typing.cast("EagerCallableProtocol", MockFXNode)("out", "output", args=(node_x,))])
    res_single = typing.cast("EagerCallableProtocol", ingest_torch_fx)(gm_single)
    assert res_single.outputs == ["x"]

    # Test output node with non-named single arg
    gm_raw = typing.cast("EagerCallableProtocol", MockFXGraphModule)([typing.cast("EagerCallableProtocol", MockFXNode)("out", "output", args=(123,))])
    res_raw = typing.cast("EagerCallableProtocol", ingest_torch_fx)(gm_raw)
    assert res_raw.outputs == []


def test_ingest_jaxpr() -> None:
    """Test ingest_jaxpr and related translation utilities."""
    with pytest.raises(ValueError, match="JAX jaxpr cannot be None"):
        ingest_jaxpr(None)

    # Empty jaxpr without eqns or consts
    res_empty = typing.cast("EagerCallableProtocol", ingest_jaxpr)(typing.cast("EagerCallableProtocol", object)())
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

    CustomObjectScope = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "CustomObjectScope"))
    deserialize_object = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "deserialize_object"))
    get_custom_objects = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "get_custom_objects"))
    get_registered_name = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "get_registered_name"))
    get_registered_object = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "get_registered_object"))
    register_serializable = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "register_serializable"))
    serialize_object = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.utils.generic_utils"), "serialize_object"))

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
    """Sample AST function returning int tuple.

    Args:
        x: Test parameter `x`.
    """
    return (11, 22, 33)


def _sample_ast_int(x: EagerValue) -> EagerValue:
    """Sample AST function returning int.

    Args:
        x: Test parameter `x`.
    """
    return 42


def _sample_ast_vars(a: EagerValue, b: EagerValue) -> EagerValue:
    """Sample AST function returning variable tuple.

    Args:
        a: Test parameter `a`.
        b: Test parameter `b`.
    """
    return (a, b)


def _sample_ast_none() -> None:
    """Sample AST function returning None."""
    return


def test_foreign_coverage() -> None:
    """Test all branches of ForeignCall.infer_shape."""
    import inspect

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

    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyOutputShapeTuple)()) == (3, 4)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyOutputShapeScalar)()) == (9,)

    # 5. First arg has output_shapes
    class DummyOutputShapesList:
        """Test class DummyOutputShapesList."""

        output_shapes = [(5, 6), (7, 8)]

    class DummyOutputShapesEmpty:
        """Test class DummyOutputShapesEmpty."""

        output_shapes = typing.cast("list[EagerTensorProtocol]", [])

    class DummyOutputShapesNonSeq:
        """Test class DummyOutputShapesNonSeq."""

        output_shapes = 123

    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyOutputShapesList)()) == (5, 6)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyOutputShapesEmpty)()) == ()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyOutputShapesNonSeq)()) == ()

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
        nodes = typing.cast("dict[str, EagerTensorProtocol]", {})

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
            """Test function __init__.

            Args:
                sg: Test parameter `sg`.
            """
            self.subgraph = sg

    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Container)(typing.cast("EagerCallableProtocol", SubgraphValid)())) == (10, 20)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Container)(typing.cast("EagerCallableProtocol", SubgraphMissingOut)())) == ()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Container)(typing.cast("EagerCallableProtocol", SubgraphNoMeta)())) == ()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Container)(typing.cast("EagerCallableProtocol", SubgraphEmptyMeta)())) == ()

    # 7. Callable first argument
    # 7a. typing.get_type_hints with return annotation
    def fn_hints(x: int) -> typing.Tuple[int, int]:
        """Test function fn_hints.

        Args:
            x: Test parameter `x`.
        """
        return (x, x)

    assert typing.cast("EagerTensorProtocol", op.infer_shape(fn_hints)) == typing.cast("EagerTensorProtocol", (int, int))

    # 7b. hints with no "return"
    def fn_no_ret(x: int) -> None:
        """Test function fn_no_ret.

        Args:
            x: Test parameter `x`.
        """
        pass

    assert op.infer_shape(fn_no_ret) == ()

    # 7c. hints with return not matching all(int or type)
    def fn_str_ret() -> typing.Tuple[str, ...]:
        """Test function fn_str_ret."""
        return ("a",)

    assert op.infer_shape(fn_str_ret) == ()

    # 7d. Callable that fails get_type_hints but signature succeeds
    class CallableWithSignature:
        """Test class CallableWithSignature."""

        def __init__(self, ret_anno: EagerValue) -> None:
            """Test function __init__.

            Args:
                ret_anno: Test parameter `ret_anno`.
            """
            self.__annotations__ = {"return": "NonExistentTypeReference"}
            self._ret_anno = ret_anno

        def __call__(self, *args: EagerValue) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
            """
            pass

    mock_callable = typing.cast("EagerCallableProtocol", CallableWithSignature)(typing.Tuple[int, int, int])
    setattr(mock_callable, "__signature__", inspect.Signature(return_annotation=typing.Tuple[int, int, int]))
    assert typing.cast("EagerTensorProtocol", op.infer_shape(mock_callable)) == typing.cast("EagerTensorProtocol", (int, int, int))

    # 7e. Callable with signature raising TypeError
    class BrokenSignatureCallable:
        """Test class BrokenSignatureCallable."""

        def __call__(self) -> None:
            """Test function __call__."""
            pass

        @property
        def __signature__(self) -> None:
            """Test function __signature__."""
            raise TypeError("broken sig")

    from ml_switcheroo_compiler.core.errors import ShapeInferenceError

    with pytest.raises(ShapeInferenceError, match="broken sig"):
        op.infer_shape(typing.cast("EagerValue", BrokenSignatureCallable()))

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

    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", PlainObj)(), typing.cast("EagerCallableProtocol", DummyTensor)()) == (1, 2)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", PlainObj)(), typing.cast("EagerCallableProtocol", PlainObj)()) == ()
