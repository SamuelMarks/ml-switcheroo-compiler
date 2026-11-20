"""Test module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.ops.control_flow.cond import cond
from ml_switcheroo_compiler.ops.control_flow.scan import scan
from ml_switcheroo_compiler.ops.control_flow.while_loop import while_loop
from tests.testing_utils import MockNodeStub


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> BackendArg:
        """Test function get_fallback_prefix.

        Returns:
            BackendArg: Computed output value from `get_fallback_prefix`.
        """
        return "bk"


class DummyVisitor(ControlFlowASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        pytest.MonkeyPatch().setattr(self, "_generator", DummyGenerator(), raising=False)


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    def __init__(self, attrs: dict[str, AttributeValue] | None = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs if attrs is not None else {}


def test_control_flow_mixin() -> None:
    """Test function test_control_flow_mixin."""
    vis = DummyVisitor()
    node = DummyNode()

    assert vis.visit_Scan(node, ["a", "b"]) == "bk_scan(a, b)"
    assert vis.visit_Switch(node, ["a", "b"]) == "bk_switch(a, b)"

    node_td = DummyNode({"wrapped_op_name": "MyOp"})
    assert vis.visit_TimeDistributed(node_td, ["a"]) == "bk_time_distributed(a, 'MyOp')"

    assert vis.visit_Assert(node, ["cond"], data=["msg"]) == "bk_assert(cond, data=['msg'])"
    assert vis.visit_Assert(node, ["cond"]) == "bk_assert(cond, data=['Assertion failed.'])"

    assert vis.visit_AssociativeScan(node, ["a", "b"]) == "bk_associative_scan(a, b)"


def test_control_flow_mixin_advanced() -> None:
    """Test function test_control_flow_mixin_advanced."""
    from ml_switcheroo_ir import IRGraph, LogicalNode

    from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor

    class DummyGenerator:
        """Test class DummyGenerator."""

        def __init__(self, graph: BackendArg = None) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph
            self.code = ["def dummy():", "    return 1"]
            self.indent_level = 0
            self.formatter = type("Formatter", (), {"indent_level": 0})()

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "bk"

        def add_line(self, line: str) -> None:
            """Test function add_line.

            Args:
                line: Test parameter `line`.
            """
            self.code.append(line)

    class DummyVisitor(ControlFlowASTVisitor):
        """Test class DummyVisitor."""

        def __init__(self) -> None:
            """Test function __init__."""
            pytest.MonkeyPatch().setattr(self, "_generator", DummyGenerator(), raising=False)
            pass

    vis = DummyVisitor()

    # Create dummy IRGraph with dummy nodes
    def make_graph() -> BackendArg:
        """Test function make_graph.

        Returns:
            BackendArg: Computed output value from `make_graph`.
        """
        g = IRGraph()
        n = LogicalNode(id="dummy_out", op_type="Identity", inputs=["x"])
        pytest.MonkeyPatch().setattr(g, "nodes", [n], raising=False)
        return g

    g_cond = make_graph()
    g_body = make_graph()
    g_true = make_graph()
    g_false = make_graph()
    g_f = make_graph()

    # visit_WhileLoop
    node_while = LogicalNode(id="while1", op_type="WhileLoop")
    node_while.attributes = ensure_attrs({"cond": g_cond, "body": g_body})

    # We must patch CodeGeneratorVisitor since it attempts to parse the graph
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.visitor.CodeGeneratorVisitor") as mock_cgv:
        mock_cgv_instance = mock_cgv.return_value
        mock_cgv_instance.generate_body.return_value = None

        res_while = vis.visit_WhileLoop(node_while, ["init_state"])
        assert res_while == "loop_val_while1"

        # visit_Cond
        node_cond = LogicalNode(id="cond1", op_type="Cond")
        node_cond.attributes = ensure_attrs({"true_branch": g_true, "false_branch": g_false})
        res_cond = vis.visit_Cond(node_cond, ["cond_val"])
        assert "if cond_val else" in res_cond

        # visit_ForiLoop
        node_fori = LogicalNode(id="fori1", op_type="ForiLoop")
        node_fori.attributes = ensure_attrs({"body": g_body})
        res_fori = vis.visit_ForiLoop(node_fori, ["0", "10", "init_val"])
        assert res_fori == "loop_val_fori1"

        # visit_Map
        node_map = LogicalNode(id="map1", op_type="Map")
        node_map.attributes = ensure_attrs({"f": g_f})
        res_map = vis.visit_Map(node_map, ["xs"])
        assert "bk.stack" in res_map

        # visit_Fold
        node_fold = LogicalNode(id="fold1", op_type="Fold")
        node_fold.attributes = ensure_attrs({"f": g_f})
        res_fold = vis.visit_Fold(node_fold, ["init", "xs"])
        assert res_fold == "fold_val_fold1"

    # visit_Vmap
    assert vis.visit_Vmap(DummyNode(), ["a", "b"]) == "bk_vmap(a, b)"

    # visit_Pmap
    assert vis.visit_Pmap(DummyNode(), ["a", "b"]) == "bk_pmap(a, b)"


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer:
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer:
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_control_flow_cond_scan_while_exhaustive() -> None:
    """Verify eager and tracing execution branches for cond, scan, and while_loop."""
    dummy_t = ensure_tensor(1.0)

    def _branch_fn() -> OpValue:
        """Return dummy tensor for conditional branch.

        Returns:
            OpValue: Dummy tensor result.
        """
        return dummy_t

    # 1. cond
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_eager", return_value="cond_eager_res") as mock_ce:
        with patch("ml_switcheroo_compiler.ops.control_flow.cond_tracing", return_value="cond_tracing_res") as mock_ct:
            config.eager_mode = True
            assert cond(dummy_t, _branch_fn, _branch_fn) == "cond_eager_res"
            mock_ce.assert_called_once()

            config.eager_mode = False
            assert cond(dummy_t, _branch_fn, _branch_fn) == "cond_tracing_res"
            mock_ct.assert_called_once()

    # 2. scan
    scan_eager_res: tuple[OpValue, Tensor] = ("scan_eager_res", dummy_t)
    scan_tracing_res: tuple[OpValue, Tensor] = ("scan_tracing_res", dummy_t)

    def _scan_step(carry: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return carry and element pair for scan step.

        Args:
            carry: Carry tensor.
            x: Element tensor.

        Returns:
            tuple[Tensor, Tensor]: Carry and output tensor pair.
        """
        return (carry, x)

    with patch("ml_switcheroo_compiler.ops.control_flow.scan_eager", return_value=scan_eager_res) as mock_se:
        with patch("ml_switcheroo_compiler.ops.control_flow.scan_tracing", return_value=scan_tracing_res) as mock_st:
            config.eager_mode = True
            assert scan(_scan_step, dummy_t, dummy_t) == scan_eager_res
            mock_se.assert_called_once()

            config.eager_mode = False
            assert scan(_scan_step, dummy_t, dummy_t) == scan_tracing_res
            mock_st.assert_called_once()

    # 3. while_loop
    def _while_cond(state: str) -> bool:
        """Return False to terminate while_loop predicate.

        Args:
            state: Loop state string.

        Returns:
            bool: Termination flag.
        """
        del state
        return False

    def _while_body(state: str) -> str:
        """Return loop state unchanged.

        Args:
            state: Loop state string.

        Returns:
            str: Loop state string.
        """
        return state

    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_eager", return_value="while_eager_res") as mock_we:
        with patch(
            "ml_switcheroo_compiler.ops.control_flow.while_loop_tracing", return_value="while_tracing_res"
        ) as mock_wt:
            config.eager_mode = True
            assert while_loop(_while_cond, _while_body, "init") == "while_eager_res"
            mock_we.assert_called_once()

            config.eager_mode = False
            assert while_loop(_while_cond, _while_body, "init") == "while_tracing_res"
            mock_wt.assert_called_once()
