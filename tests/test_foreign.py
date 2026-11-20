"""Tests for test_foreign."""

from __future__ import annotations
from tests.testing_utils import MockTestStub
from ml_switcheroo_ir import LogicalNode
from tests.testing_utils import MockTestStub
from ml_switcheroo_compiler.backends.eager_registry import BackendArg

import importlib
from collections.abc import Callable
import typing

# ruff: noqa
from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, BackendArg
from ml_switcheroo_compiler.backends.jax.jaxpr_ingest import (
    _extract_jaxpr_constants,
    _translate_jax_equation,
    ingest_jaxpr,
)
from ml_switcheroo_compiler.backends.pytorch.fx_ingest import (
    _handle_fx_call_function,
    _handle_fx_output,
    _handle_fx_placeholder,
    ingest_torch_fx,
)
from ml_switcheroo_ir import LogicalGraph
from ml_switcheroo_compiler.ops.foreign_call import ForeignCall, _coerce_foreign_dim, _extract_int_type_dims
import pytest


class MockConst(MockTestStub):
    """Mock constant."""


class MockJaxpr(MockTestStub):
    """Mock JAX program."""

    def __init__(self, consts: list[BackendArg] | None = None) -> None:
        """Initialize.

        Args:
            consts: Optional constants list.
        """
        self.consts = consts or []
        self.eqns: list[BackendArg] = []


def test_extract_jaxpr_constants() -> None:
    """Test jaxpr constant extraction."""
    graph = LogicalGraph()
    jaxpr_empty = MockJaxpr()
    _extract_jaxpr_constants(jaxpr_empty, graph)
    assert len(graph.nodes) == 0
    jaxpr_with = MockJaxpr([MockConst()])
    _extract_jaxpr_constants(jaxpr_with, graph)
    assert "const_0" in graph.nodes


def test_extract_jaxpr_constants_nodes_added() -> None:
    """Test that constants are correctly added to the graph."""
    graph = LogicalGraph()
    jaxpr_with = MockJaxpr([MockConst()])
    _extract_jaxpr_constants(jaxpr_with, graph)
    assert "const_0" in graph.nodes
    assert graph.nodes["const_0"].op_type == "Constant"


class MockFXNode(MockTestStub):
    """Mock Torch FX node."""

    def __init__(self, name: str, op: str, target: BackendArg = None, args: tuple[BackendArg, ...] = ()) -> None:
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
        """Test function __str__.

        Returns:
            str: Computed output value from `__str__`.
        """
        return self.name


class MockFXGraph:
    """Mock Torch FX graph."""

    def __init__(self, nodes: list[BackendArg]) -> None:
        """Test function __init__.

        Args:
            nodes: Test parameter `nodes`.
        """
        self.nodes = nodes


class MockFXGraphModule(MockTestStub):
    """Mock Torch FX GraphModule."""

    def __init__(self, nodes: list[BackendArg] | None = None, has_graph: bool = True) -> None:
        """Test function __init__.

        Args:
            nodes: Test parameter `nodes`.
            has_graph: Test parameter `has_graph`.
        """
        if has_graph:
            self.graph = MockFXGraph(nodes or [])


def test_ingest_torch_fx() -> None:
    """Test ingest_torch_fx under various graph configurations."""

    class _BareDummy(MockTestStub):
        """Empty dummy class without attributes."""

    with pytest.raises(ValueError, match="Torch FX GraphModule cannot be None"):
        ingest_torch_fx(None)

    # Missing graph attribute
    empty_res = ingest_torch_fx(_BareDummy())
    assert len(empty_res.nodes) == 0

    # Missing nodes attribute in graph
    missing_nodes_res = ingest_torch_fx(type("BadMod", (), {"graph": _BareDummy()})())
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
    node_add = MockFXNode("add_1", "call_function", target=TargetAdd(), args=(node_x, node_y))
    node_mul = MockFXNode("mul_1", "call_function", target=TargetMul(), args=(node_add, 2.0))
    node_unknown = MockFXNode(
        "un_1", "call_function", target="custom_sub", args=(MockFXNode("untracked", "placeholder"),)
    )
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

    class _BareDummy(MockTestStub):
        """Empty dummy class without attributes."""

    with pytest.raises(ValueError, match="JAX jaxpr cannot be None"):
        ingest_jaxpr(None)

    # Empty jaxpr without eqns or consts
    res_empty = ingest_jaxpr(_BareDummy())
    assert len(res_empty.nodes) == 0

    class MockConstWithShape:
        """Test class MockConstWithShape."""

        shape = (4, 5)

    const_var = _BareDummy()
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


def _sample_ast_tuple(x: BackendArg) -> BackendArg:
    """Sample AST function returning int tuple.

    Args:
        x: Test parameter `x`.

    Returns:
        BackendArg: Computed output value from `_sample_ast_tuple`.
    """
    return (11, 22, 33)


def _sample_ast_int(x: BackendArg) -> BackendArg:
    """Sample AST function returning int.

    Args:
        x: Test parameter `x`.

    Returns:
        BackendArg: Computed output value from `_sample_ast_int`.
    """
    return 42


def _sample_ast_vars(a: BackendArg, b: BackendArg) -> BackendArg:
    """Sample AST function returning variable tuple.

    Args:
        a: Test parameter `a`.
        b: Test parameter `b`.

    Returns:
        BackendArg: Computed output value from `_sample_ast_vars`.
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

    assert op.infer_shape(DummyOutputShapeTuple()) == (3, 4)
    assert op.infer_shape(DummyOutputShapeScalar()) == (9,)

    # 5. First arg has output_shapes
    class DummyOutputShapesList:
        """Test class DummyOutputShapesList."""

        output_shapes = [(5, 6), (7, 8)]

    class DummyOutputShapesEmpty(MockTestStub):
        """Test class DummyOutputShapesEmpty."""

        output_shapes: list[BackendArg] = []

    class DummyOutputShapesNonSeq:
        """Test class DummyOutputShapesNonSeq."""

        output_shapes = 123

    assert op.infer_shape(DummyOutputShapesList()) == (5, 6)
    assert op.infer_shape(DummyOutputShapesEmpty()) == ()
    assert op.infer_shape(DummyOutputShapesNonSeq()) == ()

    # 6. First arg has subgraph with outputs
    class NodeWithMeta:
        """Test class NodeWithMeta."""

        shape_metadata = (10, 20)

    class SubgraphValid(MockTestStub):
        """Test class SubgraphValid."""

        outputs = ["out1"]
        nodes = {"out1": NodeWithMeta()}

    class SubgraphMissingOut(MockTestStub):
        """Test class SubgraphMissingOut."""

        outputs = ["out_missing"]
        nodes: dict[str, LogicalNode] = {}

    class NodeNoMeta:
        """Test class NodeNoMeta."""

        shape_metadata = None

    class SubgraphNoMeta(MockTestStub):
        """Test class SubgraphNoMeta."""

        outputs = ["out2"]
        nodes = {"out2": NodeNoMeta()}

    class NodeEmptyMeta:
        """Test class NodeEmptyMeta."""

        shape_metadata = ()

    class SubgraphEmptyMeta(MockTestStub):
        """Test class SubgraphEmptyMeta."""

        outputs = ["out3"]
        nodes = {"out3": NodeEmptyMeta()}

    class Container(MockTestStub):
        """Test class Container."""

        def __init__(self, sg: BackendArg) -> None:
            """Test function __init__.

            Args:
                sg: Test parameter `sg`.
            """
            self.subgraph = sg

    assert op.infer_shape(Container(SubgraphValid())) == (10, 20)
    assert op.infer_shape(Container(SubgraphMissingOut())) == ()
    assert op.infer_shape(Container(SubgraphNoMeta())) == ()
    assert op.infer_shape(Container(SubgraphEmptyMeta())) == ()

    # 7. Callable first argument
    # 7a. typing.get_type_hints with return annotation
    def fn_hints(x: int) -> tuple[int, int]:
        """Test function fn_hints.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[int, int]: Computed output value from `fn_hints`.
        """
        return (x, x)

    assert op.infer_shape(fn_hints) == (int, int)

    # 7b. hints with no "return"
    def fn_no_ret(x: int) -> None:
        """Test function fn_no_ret.

        Args:
            x: Test parameter `x`.
        """
        pass

    assert op.infer_shape(fn_no_ret) == ()

    # 7c. hints with return not matching all(int or type)
    def fn_str_ret() -> tuple[str, ...]:
        """Test function fn_str_ret.

        Returns:
            tuple[str, ...]: Computed output value from `fn_str_ret`.
        """
        return ("a",)

    assert op.infer_shape(fn_str_ret) == ()

    # 7d. Callable that fails get_type_hints but signature succeeds
    class CallableWithSignature(MockTestStub):
        """Test class CallableWithSignature."""

        def __init__(self, ret_anno: BackendArg) -> None:
            """Test function __init__.

            Args:
                ret_anno: Test parameter `ret_anno`.
            """
            self.__annotations__ = {"return": "NonExistentTypeReference"}
            self._ret_anno = ret_anno

        def __call__(self, *args: BackendArg) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
            """
            pass

    mock_callable = CallableWithSignature(tuple[int, int, int])
    pytest.MonkeyPatch().setattr(
        mock_callable, "__signature__", inspect.Signature(return_annotation=tuple[int, int, int]), raising=False
    )
    assert op.infer_shape(mock_callable) == (int, int, int)

    # 7e. Callable with signature raising TypeError
    class BrokenSignatureCallable(MockTestStub):
        """Test class BrokenSignatureCallable."""

        def __call__(self) -> None:
            """Test function __call__."""
            pass

        @property
        def __signature__(self) -> None:
            """Test function __signature__.

            Raises:
                TypeError: When raised during `__signature__` (broken sig).
            """
            raise TypeError("broken sig")

    from ml_switcheroo_compiler.core.errors import ShapeInferenceError

    with pytest.raises(ShapeInferenceError, match="broken sig"):
        op.infer_shape(BrokenSignatureCallable())

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
    class PlainObj(MockTestStub):
        """Test class PlainObj."""

        pass

    assert op.infer_shape(PlainObj(), DummyTensor()) == (1, 2)
    assert op.infer_shape(PlainObj(), PlainObj()) == ()
    assert _coerce_foreign_dim(int) is int
    assert _coerce_foreign_dim("12") == 12
    assert _coerce_foreign_dim("sym") == 0
    assert _coerce_foreign_dim(None) == 0
    assert _extract_int_type_dims((3, 4)) == (3, 4)

    mock_non_int_sig = CallableWithSignature(tuple[str, int])
    pytest.MonkeyPatch().setattr(
        mock_non_int_sig, "__signature__", inspect.Signature(return_annotation=tuple[str, int]), raising=False
    )
    assert op.infer_shape(mock_non_int_sig) == ()

    non_eqn_graph = LogicalGraph()
    _translate_jax_equation("non_eqn_primitive", non_eqn_graph)
    assert non_eqn_graph.nodes["out"].op_type == "Unknown"


def test_backend_registry_ingest_graph() -> None:
    """Verify PyTorch and JAX backends register ingest_graph via BackendRegistry."""
    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    pt_backend = BackendRegistry.get("pytorch")
    jax_backend = BackendRegistry.get("jax")
    assert pt_backend is PyTorchCodeGenerator
    assert jax_backend is JAXCodeGenerator
    g_pt = PyTorchCodeGenerator.ingest_graph(MockFXGraphModule([]))
    g_jax = JAXCodeGenerator.ingest_graph(MockJaxpr())
    assert g_pt.name == "torch_fx_ingested"
    assert g_jax.name == "jaxpr_ingested"
