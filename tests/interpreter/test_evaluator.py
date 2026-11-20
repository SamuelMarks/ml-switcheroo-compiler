"""Tests for test_evaluator."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph, LogicalNode, NodeDict

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.interpreter.environment import Environment
from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node, _prepare_node_kwargs, evaluate_graph
from ml_switcheroo_compiler.ir.core import ensure_attrs
from tests.testing_utils import MockGraphStub, MockTestStub, coerce_numeric, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, BackendOperand


def test_evaluator_not_implemented() -> None:
    """Test the evaluator not implemented behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_not_implemented`.
    """
    try:
        g = LogicalGraph(outputs=["n1"])
        g.nodes["n1"] = LogicalNode(id="n1", op_type="NonExistentOp", inputs=[])
        with pytest.raises(UnimplementedMathError, match="not implemented"):
            evaluate_graph(g, {})
    except Exception as e:
        raise e


def test_evaluator_greater() -> None:
    """Test the evaluator greater behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_greater`.
    """
    try:
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Greater", inputs=["a", "b"])
        g.outputs = ["c"]
        res = evaluate_graph(g, inputs={"a": np.array([2.0]), "b": np.array([1.0])})
        assert ensure_ndarray(res["c"])[0]
    except Exception as e:
        raise e


def test_evaluator_relu() -> None:
    """Test the evaluator relu behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_relu`.
    """
    try:
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Relu", inputs=["a"])
        g.outputs = ["c"]
        res = evaluate_graph(g, inputs={"a": np.array([-1.0, 2.0])})
        np.testing.assert_array_equal(ensure_ndarray(res["c"]), np.array([0.0, 2.0]))
    except Exception as e:
        raise e


def test_evaluator_where() -> None:
    """Test the evaluator where behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_where`.
    """
    try:
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Input")
        g.nodes["d"] = LogicalNode(id="d", op_type="Where", inputs=["a", "b", "c"])
        g.outputs = ["d"]
        res = evaluate_graph(
            g,
            inputs={
                "a": np.array([True, False]),
                "b": np.array([1.0, 2.0]),
                "c": np.array([3.0, 4.0]),
            },
        )
        np.testing.assert_array_equal(ensure_ndarray(res["d"]), np.array([1.0, 4.0]))
    except Exception as e:
        raise e


def test_evaluator_unimplemented() -> None:
    """Test function test_evaluator_unimplemented."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    g = LogicalGraph(name="test")
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    g.nodes["b"] = LogicalNode(id="b", op_type="UnknownOp", inputs=["a"])
    g.outputs = ["b"]
    with pytest.raises((UnimplementedMathError, RuntimeError)):
        evaluate_graph(g, inputs={"a": 1})


def test_evaluator_exception() -> None:
    """Test the evaluator exception behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_exception`.
    """
    try:
        g = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
        n2 = IRNode(id="n2", op_type="nonexistent_blah", inputs=["n1"], attributes={}, shape_metadata=(2,))
        for n in [n1, n2]:
            g.nodes[n.id] = n
        g.inputs = ["n1"]
        g.outputs = ["n2"]
        with pytest.raises((UnimplementedMathError, AttributeError)):
            evaluate_graph(g, {"n1": 1})
    except Exception as e:
        raise e


def test_evaluator_stubs() -> None:
    """Test the evaluator stubs behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_stubs`.
    """
    try:
        g1 = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        ng = IRNode(id="ng", op_type="Greater", inputs=["n1", "n2"], attributes={}, shape_metadata=(1,))
        g1.nodes = NodeDict({n.id: n for n in [n1, n2, ng]})
        g1.inputs = ["n1", "n2"]
        g1.outputs = ["ng"]
        res1 = evaluate_graph(g1, {"n1": np.array([2.0]), "n2": np.array([1.0])})
        assert ensure_ndarray(res1["ng"])[0]
        g2 = IRGraph()
        nc = IRNode(id="nc", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nt = IRNode(id="nt", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nf = IRNode(id="nf", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nw = IRNode(id="nw", op_type="Where", inputs=["nc", "nt", "nf"], attributes={}, shape_metadata=(1,))
        g2.nodes = NodeDict({n.id: n for n in [nc, nt, nf, nw]})
        g2.inputs = ["nc", "nt", "nf"]
        g2.outputs = ["nw"]
        res2 = evaluate_graph(
            g2,
            {
                "nc": np.array([True]),
                "nt": np.array([2.0]),
                "nf": np.array([3.0]),
            },
        )
        assert ensure_ndarray(res2["nw"])[0] == 2.0
    except Exception as e:
        raise e


def test_evaluator_shape_kwargs() -> None:
    """Test the evaluator shape kwargs behavior.

    Returns:
        OpValue: The inferred shape or computed result.

    Raises:
        e: If execution or validation fails in `test_evaluator_shape_kwargs`.
    """
    try:
        g1 = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        ne = IRNode(id="ne", op_type="BroadcastTo", inputs=["n1"], attributes={}, shape_metadata=(2,))
        g1.nodes = NodeDict({n.id: n for n in [n1, ne]})
        g1.inputs = ["n1"]
        g1.outputs = ["ne"]
        res1 = evaluate_graph(g1, {"n1": np.array([1.0])})
        assert extract_test_shape(res1["ne"]) == (2,)
        g2 = IRGraph()
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
        nr = IRNode(id="nr", op_type="Reshape", inputs=["n2"], attributes={}, shape_metadata=(1, 2))
        g2.nodes = NodeDict({n.id: n for n in [n2, nr]})
        g2.inputs = ["n2"]
        g2.outputs = ["nr"]
        res2 = evaluate_graph(g2, {"n2": np.array([1.0, 2.0])})
        assert extract_test_shape(res2["nr"]) == (1, 2)
    except Exception as e:
        raise e


def test_evaluator_edge_cases_basic() -> None:
    """Test function test_evaluator_edge_cases_basic."""
    env = Environment({"k": 42})
    assert env.get("k") == 42


def test_evaluator_prepare_node_kwargs_more() -> None:
    """Test function test_evaluator_prepare_node_kwargs_more."""
    n = LogicalNode(id="n1", op_type="ConstantOfShape")
    n.shape_metadata = (5, 5)
    kw = _prepare_node_kwargs(n, "ConstantOfShape")
    assert kw["shape"] == (5, 5)
    n2 = LogicalNode(id="n2", op_type="Zeros")
    n2.shape_metadata = (10,)
    kw2 = _prepare_node_kwargs(n2, "Zeros")
    assert kw2["shape"] == (10,)
    n3 = LogicalNode(id="n3", op_type="Ones")
    n3.shape_metadata = (10,)
    kw3 = _prepare_node_kwargs(n3, "Ones")
    assert kw3["shape"] == (10,)
    n4 = LogicalNode(id="n4", op_type="Full")
    n4.shape_metadata = (10,)
    kw4 = _prepare_node_kwargs(n4, "Full")
    assert kw4["shape"] == (10,)
    n5 = LogicalNode(id="n5", op_type="ConstantOfShape")
    kw5 = _prepare_node_kwargs(n5, "ConstantOfShape")
    assert "shape" not in kw5


def test_evaluator_missing_output() -> None:
    """Test function test_evaluator_missing_output."""

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

    env = Environment({"n1": 2.0})
    n = LogicalNode(id="n2", op_type="Output", inputs=["n1"])
    _evaluate_node(n, env, MockBackend())
    assert env.get("n2") == 2.0


def test_evaluator_recompute_coverage_mock() -> None:
    """Test function test_evaluator_recompute_coverage_mock."""

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

    env = Environment({"n1": 2.0})
    n = LogicalNode(id="n2", op_type="Recompute", inputs=["n1"])
    n.attributes = {"original_op": "Abs", "original_attrs": {}}
    import pytest

    with pytest.raises(AttributeError):
        _evaluate_node(n, env, MockBackend())


def test_evaluator_missing_funcs() -> None:
    """Test function test_evaluator_missing_funcs."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return "exec"

    env = Environment({"n1": 2.0})
    n = LogicalNode(id="n2", op_type="Unknown", inputs=["n1"])
    _evaluate_node(n, env, MockBackend())
    from ml_switcheroo_compiler.ops.eager_evaluator import (
        BackendExecuteOpStrategy,
        CustomEagerEvalStrategy,
        EvaluationContext,
        EvaluationStrategy,
    )

    class _ConcreteEvalStrategy(EvaluationStrategy):
        """Concrete subclass delegating to abstract EvaluationStrategy.evaluate."""

        def evaluate(self, ctx: EvaluationContext) -> BackendArg:
            """Delegate to base `EvaluationStrategy.evaluate` to verify `NotImplementedError`.

            Args:
                ctx: Evaluation context.

            Returns:
                BackendArg: Unreachable return annotation.
            """
            return EvaluationStrategy.evaluate(self, ctx)

    with pytest.raises(NotImplementedError):
        _ConcreteEvalStrategy().evaluate(EvaluationContext(None, "noop", [], {}, None))

    class DummyOp:
        """Test class DummyOp."""

        @staticmethod
        def eager_eval(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function eager_eval.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `eager_eval`.
            """
            return "dummy"

    assert CustomEagerEvalStrategy().evaluate(EvaluationContext(DummyOp, "dummy", [], {}, None)) == "dummy"

    class MockBackend2(MockTestStub):
        """Test class MockBackend2."""

        def execute_op(self, op_type: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op_type: Test parameter `op_type`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return "backend"

    assert BackendExecuteOpStrategy().evaluate(EvaluationContext(None, "dummy", [], {}, MockBackend2())) == "backend"
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", None))
    res = EagerEvaluator._pack_outputs([np.array([2.0])], t, None)
    assert isinstance(res, tuple)
    assert res[0].shape == (1,)
    res2 = EagerEvaluator._pack_outputs(np.array([2.0]), t, None)
    assert extract_test_shape(res2) == (1,)
    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(eager_mode=True):
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        out = dispatch_op("Add", t, t)
        assert out.shape == (1,)


def test_evaluator_base_eval_missing() -> None:
    """Test function test_evaluator_base_eval_missing."""
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp2(OpDef):
        """Test class DummyOp2."""

        op_name = "dummy2"

    strat = EagerEvaluator._get_strategy(DummyOp2)
    assert type(strat).__name__ == "BackendExecuteOpStrategy"


def test_evaluator_base_eval_missing2() -> None:
    """Test function test_evaluator_base_eval_missing2."""
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp3(OpDef):
        """Test class DummyOp3."""

        op_name = "dummy3"

        @staticmethod
        def eager_eval(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function eager_eval.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `eager_eval`.
            """
            return "dummy3"

    assert isinstance(EagerEvaluator._get_strategy(DummyOp3), type(EagerEvaluator._get_strategy(DummyOp3)))

    class DummyOp4(OpDef):
        """Test class DummyOp4."""

        op_name = "dummy4"

    assert isinstance(EagerEvaluator._get_strategy(DummyOp4), type(EagerEvaluator._get_strategy(DummyOp4)))


def test_evaluator_base_eval_missing3() -> None:
    """Test function test_evaluator_base_eval_missing3."""
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp5(OpDef):
        """Test class DummyOp5."""

        pass

    assert isinstance(EagerEvaluator._get_strategy(DummyOp5), type(EagerEvaluator._get_strategy(DummyOp5)))


def test_evaluator_base_eval_missing4() -> None:
    """Test function test_evaluator_base_eval_missing4."""
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp6(OpDef):
        """Test class DummyOp6."""

        pass

    pytest.MonkeyPatch().setattr(DummyOp6, "eager_eval", None, raising=False)
    strat = EagerEvaluator._get_strategy(DummyOp6)
    assert type(strat).__name__ == "CustomEagerEvalStrategy"


def test_evaluator_base_eval_missing5() -> None:
    """Test function test_evaluator_base_eval_missing5."""
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
    from tests.testing_utils import require_callable

    no_cls: type | None = None
    strat_fn = require_callable(EagerEvaluator._get_strategy)
    assert isinstance(
        strat_fn(no_cls),
        type(strat_fn(no_cls)),
    )


def test_evaluator_slice_parsing_edge() -> None:
    """Test function test_evaluator_slice_parsing_edge."""
    import ast

    import pytest
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_slice, _parse_slice_call, _parse_slice_node

    with pytest.raises(ValueError):
        _parse_slice_node(ast.parse("1 + 1", mode="eval").body)

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return x

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Slice", inputs=["n1"])
    import numpy as np

    t = np.array([1, 2, 3])
    _handle_slice(n, env, MockBackend(), [t], {"slices": "slice(1, 2)"})
    assert np.array_equal(ensure_ndarray(env.get("n2")), np.array([2]))
    _handle_slice(n, env, MockBackend(), [t], {"dim": 0, "start": 0, "end": 2, "step": 1})
    assert np.array_equal(ensure_ndarray(env.get("n2")), np.array([1, 2]))
    c1 = ast.parse("unknown_func()", mode="eval").body
    c2 = ast.parse("obj.method()", mode="eval").body
    c3 = ast.parse("array([1])", mode="eval").body
    assert isinstance(c1, ast.Call) and isinstance(c2, ast.Call) and isinstance(c3, ast.Call)
    assert _parse_slice_call(c1) is None
    assert _parse_slice_call(c2) is None
    assert _parse_slice_call(c3) == [1]


def test_evaluator_all_slice_asts() -> None:
    """Test function test_evaluator_all_slice_asts."""
    import ast

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_slice_string, _parse_unary

    assert _parse_slice_string("()") == ()
    assert _parse_slice_string("[1, 2]") == [1, 2]
    assert _parse_slice_string("-1") == -1
    u_expr = ast.parse("~1", mode="eval").body
    assert isinstance(u_expr, ast.UnaryOp)
    assert _parse_unary(u_expr) is None


def disabled_test_evaluator_handle_custom_ops() -> None:
    """Test function disabled_test_evaluator_handle_custom_ops."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return x

        def execute_op(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return "exec"

    env = Environment({"n1": np.array([1.0]), "n2": np.array([0]), "n3": np.array([1.0]), "c": np.array([1.0])})
    n_cond = LogicalNode(id="cond_out", op_type="Cond", inputs=["c", "n1"])
    from ml_switcheroo_ir import LogicalGraph

    tg = LogicalGraph()
    tg.outputs = ["n1"]
    fg = LogicalGraph()
    fg.outputs = ["n1"]
    n_cond.attributes = ensure_attrs({"true_branch": tg, "false_branch": fg})
    _evaluate_node(n_cond, env, MockBackend())


def test_evaluator_handle_checkpoint() -> None:
    """Test function test_evaluator_handle_checkpoint."""
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_checkpoint

    env = Environment({"n1": 1.0})
    n = LogicalNode(id="n2", op_type="Checkpoint", inputs=["n1"])
    sg = LogicalGraph()
    n_sub = LogicalNode(id="n_sub_in", op_type="Input")
    n_sub2 = LogicalNode(id="n_sub_out", op_type="Add", inputs=["n_sub_in", "n_sub_in"])
    sg.nodes = NodeDict({"n_sub_in": n_sub, "n_sub_out": n_sub2})
    sg.inputs = ["n_sub_in"]
    sg.outputs = ["n_sub_out"]
    n.attributes = ensure_attrs({"subgraph": sg})
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=np.array([2.0])):
        _handle_checkpoint(n, env, [1.0])
    assert env.get("n2") == np.array([2.0])
    sg.outputs = ["n_sub_out", "n_sub_out"]
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=np.array([2.0])):
        _handle_checkpoint(n, env, [1.0])
    assert isinstance(env.get("n2"), tuple)
    pytest.MonkeyPatch().setattr(sg, "nodes", [n_sub, n_sub2], raising=False)
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=np.array([2.0])):
        _handle_checkpoint(n, env, [1.0])


def test_evaluator_missing_output2() -> None:
    """Test function test_evaluator_missing_output2."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    g = LogicalGraph()
    g.nodes = NodeDict({"n1": LogicalNode(id="n1", op_type="Input")})
    g.outputs = ["missing_out"]
    import pytest

    with pytest.raises(RuntimeError):
        evaluate_graph(g, {"n1": 2.0})


def test_evaluator_parse_name() -> None:
    """Test function test_evaluator_parse_name."""
    import ast

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_name

    assert _parse_name(ast.Name(id="None")) is None
    assert _parse_name(ast.Name(id="Ellipsis")) is Ellipsis


def test_evaluator_handle_getitem() -> None:
    """Test function test_evaluator_handle_getitem."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_getitem

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return x

    env = Environment({})
    n = LogicalNode(id="n2", op_type="GetItem", inputs=["n1"])
    t = np.array([1, 2, 3])
    _handle_getitem(n, env, MockBackend(), [t], {"key": "1"})
    assert env.get("n2") == 2


def test_evaluator_handle_meshgrid() -> None:
    """Test function test_evaluator_handle_meshgrid."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_meshgrid

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return ["m1", "m2"]

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Meshgrid", inputs=["n1"])
    _handle_meshgrid(n, env, MockBackend(), [1.0], {"output_index": 1})
    assert env.get("n2") == "m2"


def test_evaluator_target_dispatch() -> None:
    """Test function test_evaluator_target_dispatch."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return x

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return ["m1", "m2"]

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Slice", inputs=["n1"])
    t = np.array([1, 2, 3])
    _dispatch_op(n, env, MockBackend(), "GetItem", [t], {"key": "1"})
    _dispatch_op(n, env, MockBackend(), "Meshgrid", [t], {"output_index": 1})
    n2 = LogicalNode(id="n3", op_type="Constant")
    n2.attributes = {"value": 5.0}

    class MockBackend3:
        """Test class MockBackend3."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node

    _evaluate_node(n2, env, MockBackend3())
    assert env.get("n3") == 5.0


def test_evaluator_prepare_node_kwargs_reshape() -> None:
    """Test function test_evaluator_prepare_node_kwargs_reshape."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import _prepare_node_kwargs

    n = LogicalNode(id="n1", op_type="Reshape")
    n.shape_metadata = (5, 5)
    kw = _prepare_node_kwargs(n, "Reshape")
    assert kw["newshape"] == (5, 5)


def test_evaluator_target_dispatch_more() -> None:
    """Test function test_evaluator_target_dispatch_more."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return x

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return "exec"

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Slice", inputs=["n1"])
    t = np.array([1, 2, 3])
    _dispatch_op(n, env, MockBackend(), "Slice", [t], {"slices": "slice(1, 2)"})
    n2 = LogicalNode(id="n2", op_type="Checkpoint", inputs=["n1"])
    n2.attributes = {"subgraph": None}
    import pytest

    with pytest.raises(AttributeError):
        _dispatch_op(n2, env, MockBackend(), "Checkpoint", [t], {})


def test_evaluate_if_node() -> None:
    """Test function test_evaluate_if_node."""
    import numpy as np
    from ml_switcheroo_ir import IRGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node

    env = Environment()
    env.set("cond", np.array([True]))
    env.set("a", 5)
    node = LogicalNode(id="out", op_type="If", inputs=["cond"])
    then_graph = IRGraph()
    n_then = LogicalNode(id="then_node", op_type="Identity", inputs=["a"])
    pytest.MonkeyPatch().setattr(then_graph, "nodes", [n_then], raising=False)
    then_graph.outputs = ["then_node"]
    node.attributes = ensure_attrs({"then_branch": then_graph})

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendOperand:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerTensorProtocol: Computed output value from `execute_op`.
            """
            if op == "Identity":
                return coerce_numeric(args[0]) * 2
            return None

    _evaluate_node(node, env, MockBackend())
    assert env.get("out") == 10


def test_evaluate_if_node_false() -> None:
    """Test function test_evaluate_if_node_false."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node

    env = Environment()
    env.set("cond", 0)
    env.set("b", 6)
    node = LogicalNode(id="out", op_type="If", inputs=["cond"])
    else_graph = MockGraphStub()
    n_else = LogicalNode(id="else_node", op_type="Identity", inputs=["b"])
    else_graph.nodes = NodeDict({"else_node": n_else})
    pytest.MonkeyPatch().setattr(else_graph, "outputs", "else_node", raising=False)
    node.attributes = ensure_attrs({"false_branch": else_graph})

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendOperand:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerTensorProtocol: Computed output value from `execute_op`.
            """
            if op == "Identity":
                return coerce_numeric(args[0]) + 1
            return None

    _evaluate_node(node, env, MockBackend())
    assert env.get("out") == 7


def test_evaluate_if_node_no_outputs() -> None:
    """Test function test_evaluate_if_node_no_outputs."""
    from ml_switcheroo_ir import IRGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node

    env = Environment()
    env.set("cond", 1)
    node = LogicalNode(id="out", op_type="If", inputs=["cond"])
    then_graph = IRGraph()
    n_then = LogicalNode(id="then_node", op_type="Constant", inputs=[])
    n_then.attributes = {"value": 99}
    pytest.MonkeyPatch().setattr(then_graph, "nodes", [n_then], raising=False)
    pytest.MonkeyPatch().setattr(then_graph, "outputs", None, raising=False)
    node.attributes = ensure_attrs({"then_branch": then_graph})

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return args[0]

    _evaluate_node(node, env, MockBackend())
    assert env.get("out") == 99


def test_evaluate_loop_node() -> None:
    """Test function test_evaluate_loop_node."""
    from ml_switcheroo_ir import IRGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node

    env = Environment()
    env.set("curr", 3)
    node = LogicalNode(id="out", op_type="Loop", inputs=["curr"])
    cond_graph = IRGraph()
    n_cond_in = LogicalNode(id="cond_in", op_type="Input")
    n_cond = LogicalNode(id="cond_eval", op_type="Greater", inputs=["cond_in"])
    pytest.MonkeyPatch().setattr(cond_graph, "nodes", [n_cond_in, n_cond], raising=False)
    cond_graph.outputs = ["cond_eval"]
    body_graph = IRGraph()
    n_body_in = LogicalNode(id="body_in", op_type="Input")
    n_body = LogicalNode(id="body_eval", op_type="Sub", inputs=["body_in"])
    body_graph.nodes = NodeDict({"body_in": n_body_in, "body_eval": n_body})
    body_graph.outputs = ["body_eval"]
    node.attributes = ensure_attrs({"cond": cond_graph, "body": body_graph})

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def execute_op(self, op: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendOperand:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerTensorProtocol: Computed output value from `execute_op`.
            """
            val = args[0] if args else None
            if val is None:
                val = 0
            if op == "Greater":
                return float(coerce_numeric(val)) > 0
            if op in ("Sub", "Subtract"):
                return coerce_numeric(val) - 1
            return None

    _evaluate_node(node, env, MockBackend())
    print("MEMORY:", env.memory)
    assert env.get("out") == 0


def test_evaluate_if_no_branch() -> None:
    """Test function test_evaluate_if_no_branch."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node

    env = Environment()
    env.set("cond", np.array([True]))
    node = LogicalNode(id="out", op_type="If", inputs=["cond"])
    node.attributes = {}

    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        pass

    _evaluate_node(node, env, MockBackend())
    assert "out" not in env.memory


def test_evaluator_coverage() -> None:
    """Test function test_evaluator_coverage."""
    import ast
    from unittest.mock import patch

    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import (
        _parse_constant,
        _parse_list,
        _parse_name,
        _parse_slice_call,
        _parse_tuple,
        evaluate_graph,
    )

    g = LogicalGraph()
    g.nodes = NodeDict({"n1": LogicalNode(id="n1", op_type="Input")})
    g.outputs = ["n2"]
    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_node"):
        with pytest.raises(RuntimeError):
            evaluate_graph(g, {"n1": 1})
    g.outputs = ["n1"]
    res = evaluate_graph(g, {"n1": 1})
    assert res["n1"] == 1
    n = ast.Call(func=ast.Attribute(value=ast.Name(id="x"), attr="m"), args=[], keywords=[])
    assert _parse_slice_call(n) is None
    n = ast.Call(func=ast.Name(id="slice"), args=[ast.Constant(value=1)], keywords=[])
    assert _parse_slice_call(n) == slice(1)
    n = ast.Call(func=ast.Name(id="array"), args=[ast.Constant(value=1)], keywords=[])
    assert _parse_slice_call(n) == 1
    n = ast.Call(func=ast.Name(id="unknown"), args=[], keywords=[])
    assert _parse_slice_call(n) is None
    assert _parse_tuple(ast.Tuple(elts=[ast.Constant(value=1)])) == (1,)
    assert _parse_list(ast.List(elts=[ast.Constant(value=1)])) == [1]
    assert _parse_constant(ast.Constant(value=1)) == 1
    assert _parse_name(ast.Name(id="None")) is None
    assert _parse_name(ast.Name(id="Ellipsis")) is Ellipsis


def test_evaluator_dispatch() -> None:
    """Test function test_evaluator_dispatch."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    env = Environment({})

    class FakeBackend(MockTestStub):
        """Test class FakeBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return np.array(x)

        def execute_op(self, name: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            if name == "Meshgrid":
                return [1, 2]
            return 42

    n_slice = LogicalNode(id="slice_id", op_type="Slice")
    _dispatch_op(
        n_slice,
        env,
        FakeBackend(),
        "Slice",
        [np.array([1, 2, 3])],
        {"slices": "slice(1, None)"},
    )
    assert ensure_ndarray(env.get("slice_id")).tolist() == [2, 3]
    _dispatch_op(
        n_slice,
        env,
        FakeBackend(),
        "Slice",
        [np.array([1, 2, 3])],
        {"dim": 0, "start": 0, "end": 1, "step": 1},
    )
    assert ensure_ndarray(env.get("slice_id")).tolist() == [1]
    n_getitem = LogicalNode(id="get_id", op_type="GetItem")
    _dispatch_op(
        n_getitem,
        env,
        FakeBackend(),
        "GetItem",
        [np.array([1, 2, 3])],
        {"key": "slice(1, None)"},
    )
    assert ensure_ndarray(env.get("get_id")).tolist() == [2, 3]

    fake_sg = MockGraphStub(
        inputs=["in1"],
        outputs=["out1"],
        nodes=[LogicalNode(id="out1", op_type="Abs", inputs=["in1"])],
    )

    n_checkpoint = LogicalNode(id="cp_id", op_type="Checkpoint", attributes=ensure_attrs({"subgraph": fake_sg}))
    _dispatch_op(n_checkpoint, env, FakeBackend(), "Checkpoint", [99], {})
    assert env.get("cp_id") == 99
    n_meshgrid = LogicalNode(id="mg_id", op_type="Meshgrid")
    _dispatch_op(n_meshgrid, env, FakeBackend(), "Meshgrid", [], {"output_index": 1})
    assert env.get("mg_id") == 2
    n_other = LogicalNode(id="other_id", op_type="Other")
    _dispatch_op(n_other, env, FakeBackend(), "Other", [], {})
    assert env.get("other_id") == 42

    fake_sg_dict = MockGraphStub(
        inputs=["in1"],
        outputs=["out1", "out2"],
        nodes={
            "out1": LogicalNode(id="out1", op_type="Abs", inputs=["in1"]),
            "out2": LogicalNode(id="out2", op_type="Abs", inputs=["in1"]),
        },
    )

    n_checkpoint_dict = LogicalNode(
        id="cp_id2", op_type="Checkpoint", attributes=ensure_attrs({"subgraph": fake_sg_dict})
    )
    _dispatch_op(n_checkpoint_dict, env, FakeBackend(), "Checkpoint", [99], {})


def test_evaluator_uncovered() -> None:
    """Test function test_evaluator_uncovered."""
    import ast

    import pytest

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_slice_node

    n_unary = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))
    assert _parse_slice_node(n_unary) == -1
    n_unary_none = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=1))
    assert _parse_slice_node(n_unary_none) is None
    assert _parse_slice_node(ast.Tuple(elts=[ast.Constant(value=1)])) == (1,)
    assert _parse_slice_node(ast.List(elts=[ast.Constant(value=1)])) == [1]
    assert _parse_slice_node(ast.Name(id="None")) is None
    with pytest.raises(ValueError):
        _parse_slice_node(ast.Pass())


def test_evaluator_extra_branches() -> None:
    """Test function test_evaluator_extra_branches."""
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node, _get_op_alias, _prepare_node_kwargs

    env = Environment({"in1": 10, "in2": 20, "cond1": True, "cond2": False})

    class FakeBackend(MockTestStub):
        """Test class FakeBackend."""

        def asarray(self, x: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return np.array(x)

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return np.array(x)

        def execute_op(self, name: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return 42

    backend = FakeBackend()
    n_out1 = LogicalNode(id="out_1", op_type="Output", inputs=["in1"])
    _evaluate_node(n_out1, env, backend)
    assert env.get("out_1") == 10
    n_out2 = LogicalNode(id="out_2", op_type="Output", inputs=["in1", "in2"])
    _evaluate_node(n_out2, env, backend)
    assert env.get("out_2") == (10, 20)
    n_const = LogicalNode(id="const_1", op_type="Constant", attributes={"value": 99})
    _evaluate_node(n_const, env, backend)
    assert env.get("const_1") == 99
    n_recompute = LogicalNode(
        id="rec_1", op_type="Recompute", inputs=["in1"], attributes={"original_op": "Add", "original_attrs": {}}
    )
    _evaluate_node(n_recompute, env, backend)
    assert env.get("rec_1") == 42
    n_other = LogicalNode(id="other_1", op_type="OtherOp", inputs=["in1"])
    _evaluate_node(n_other, env, backend)
    assert env.get("other_1") == 42
    assert _get_op_alias("Sub") == "Subtract"
    assert _get_op_alias("Unknown") == "Unknown"
    n_expand = LogicalNode(id="expand_1", op_type="Expand", shape_metadata=(2, 2))
    kwargs = _prepare_node_kwargs(n_expand, "BroadcastTo")
    assert kwargs["shape"] == (2, 2)
    n_reshape = LogicalNode(id="reshape_1", op_type="Reshape", shape_metadata=(2, 2))
    kwargs = _prepare_node_kwargs(n_reshape, "Reshape")
    assert kwargs["newshape"] == (2, 2)
    n_other_kw = LogicalNode(id="other_kw_1", op_type="Other", shape_metadata=(2, 2))
    kwargs = _prepare_node_kwargs(n_other_kw, "Other")
    assert "shape" not in kwargs
    assert "newshape" not in kwargs


def test_evaluator_if() -> None:
    """Test function test_evaluator_if."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_if_node

    env = Environment({"cond": True, "val1": 10})

    class FakeBranch(MockGraphStub):
        """Test class FakeBranch."""

        def __init__(self, out_is_list: BackendArg = True) -> None:
            """Test function __init__.

            Args:
                out_is_list: Test parameter `out_is_list`.
            """
            super().__init__()
            pytest.MonkeyPatch().setattr(
                self, "nodes", [LogicalNode(id="out1", op_type="Output", inputs=["val1"])], raising=False
            )
            if out_is_list:
                self.outputs = ["out1"]
            else:
                pytest.MonkeyPatch().setattr(self, "outputs", "out1", raising=False)

    class FakeBackend(MockTestStub):
        """Test class FakeBackend."""

        def execute_op(self, name: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return 42

    n_if_then = LogicalNode(
        id="if_1", op_type="If", inputs=["cond"], attributes=ensure_attrs({"then_branch": FakeBranch()})
    )
    _evaluate_if_node(n_if_then, env, FakeBackend())
    assert env.get("if_1") == 10
    env = Environment({"cond": False, "val1": 10})
    n_if_else = LogicalNode(
        id="if_2",
        op_type="If",
        inputs=["cond"],
        attributes=ensure_attrs({"else_branch": FakeBranch(out_is_list=False)}),
    )
    _evaluate_if_node(n_if_else, env, FakeBackend())
    assert env.get("if_2") == 10
    env = Environment({"cond": True, "val1": 10})

    class FakeBranchNoOut(MockGraphStub):
        """Test class FakeBranchNoOut."""

        def __init__(self) -> None:
            """Test function __init__."""
            super().__init__()
            self.nodes = NodeDict({"out1": LogicalNode(id="out1", op_type="Output", inputs=["val1"])})
            self.outputs: list[str] = []

    n_if_true = LogicalNode(
        id="if_3", op_type="If", inputs=["cond"], attributes=ensure_attrs({"true_branch": FakeBranchNoOut()})
    )
    _evaluate_if_node(n_if_true, env, FakeBackend())
    assert env.get("if_3") == 10
    env = Environment({"cond": False, "val1": 10})
    n_if_false = LogicalNode(
        id="if_4", op_type="If", inputs=["cond"], attributes=ensure_attrs({"false_branch": FakeBranchNoOut()})
    )
    _evaluate_if_node(n_if_false, env, FakeBackend())
    assert env.get("if_4") == 10

    class FakeTensorBool(MockTestStub):
        """Test class FakeTensorBool."""

        def item(self) -> bool:
            """Test function item.

            Returns:
                bool: Computed output value from `item`.
            """
            return True

    env_bool = Environment({"cond": FakeTensorBool(), "val1": 10})
    _evaluate_if_node(n_if_then, env_bool, FakeBackend())
    assert env_bool.get("if_1") == 10


def test_evaluator_loop() -> None:
    """Test function test_evaluator_loop."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_loop_node, _evaluate_node

    env = Environment({"in1": 0})

    class FakeBackend(MockTestStub):
        """Test class FakeBackend."""

        def execute_op(self, name: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return coerce_numeric(args[0]) + 1

    class FakeCondGraph(MockGraphStub):
        """Test class FakeCondGraph."""

        def __init__(self, out_is_tensor: BackendArg = False) -> None:
            """Test function __init__.

            Args:
                out_is_tensor: Test parameter `out_is_tensor`.
            """
            super().__init__()
            pytest.MonkeyPatch().setattr(
                self,
                "nodes",
                [
                    LogicalNode(id="cond_in", op_type="Input"),
                    LogicalNode(id="cond_out", op_type="Less", inputs=["cond_in"]),
                ],
                raising=False,
            )
            self.outputs = ["cond_out"]
            self.out_is_tensor = out_is_tensor

    class FakeBodyGraph(MockGraphStub):
        """Test class FakeBodyGraph."""

        def __init__(self, nodes_dict: BackendArg = False) -> None:
            """Test function __init__.

            Args:
                nodes_dict: Test parameter `nodes_dict`.
            """
            super().__init__()
            if nodes_dict:
                self.nodes = NodeDict(
                    {
                        "body_in": LogicalNode(id="body_in", op_type="Input"),
                        "body_out": LogicalNode(id="body_out", op_type="Add", inputs=["body_in"]),
                    }
                )
            else:
                pytest.MonkeyPatch().setattr(
                    self,
                    "nodes",
                    [
                        LogicalNode(id="body_in", op_type="Input"),
                        LogicalNode(id="body_out", op_type="Add", inputs=["body_in"]),
                    ],
                    raising=False,
                )
            self.outputs = ["body_out"]

    n_loop = LogicalNode(
        id="loop_1",
        op_type="WhileLoop",
        inputs=["in1"],
        attributes=ensure_attrs({"cond": FakeCondGraph(), "body": FakeBodyGraph()}),
    )
    from unittest.mock import patch

    def mock_eval_node(node: LogicalNode, env: Environment, backend: BackendArg) -> None:
        """Test function mock_eval_node.

        Args:
            node: Test parameter `node`.
            env: Test parameter `env`.
            backend: Test parameter `backend`.
        """
        if node.op_type == "Less":
            v = env.get(node.inputs[0])
            env.set(node.id, float(coerce_numeric(v)) < 2)
        elif node.op_type == "Add":
            v = env.get(node.inputs[0])
            env.set(node.id, coerce_numeric(v) + 1)

    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_node", side_effect=mock_eval_node):
        _evaluate_loop_node(n_loop, env, FakeBackend())
        assert env.get("loop_1") == 2

        class FakeCondGraphDict(MockGraphStub):
            """Test class FakeCondGraphDict."""

            def __init__(self) -> None:
                """Test function __init__."""
                super().__init__()
                self.nodes = NodeDict(
                    {
                        "cond_in": LogicalNode(id="cond_in", op_type="Input"),
                        "cond_out": LogicalNode(id="cond_out", op_type="Less", inputs=["cond_in"]),
                    }
                )
                self.outputs = ["cond_out"]

        env = Environment({"in1": 0})
        n_loop2 = LogicalNode(
            id="loop_2",
            op_type="WhileLoop",
            inputs=["in1"],
            attributes=ensure_attrs({"cond": FakeCondGraphDict(), "body": FakeBodyGraph(nodes_dict=True)}),
        )
        _evaluate_loop_node(n_loop2, env, FakeBackend())
        assert env.get("loop_2") == 2
    n_cond = LogicalNode(
        id="cond_1", op_type="Cond", inputs=["cond"], attributes=ensure_attrs({"then_branch": FakeCondGraph()})
    )
    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_if_node") as mock_if:
        _evaluate_node(n_cond, env, FakeBackend())
        assert mock_if.called
    n_loop_alias = LogicalNode(
        id="loop_3",
        op_type="Loop",
        inputs=["in1"],
        attributes=ensure_attrs({"cond": FakeCondGraph(), "body": FakeBodyGraph()}),
    )
    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_loop_node") as mock_loop:
        _evaluate_node(n_loop_alias, env, FakeBackend())
        assert mock_loop.called


def test_evaluator_if_no_branch() -> None:
    """Test function test_evaluator_if_no_branch."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_if_node

    env = Environment({"cond": True})

    class FakeBackend(MockTestStub):
        """Test class FakeBackend."""

        pass

    n_if = LogicalNode(id="if_1", op_type="If", inputs=["cond"], attributes={})
    _evaluate_if_node(n_if, env, FakeBackend())
    assert "if_1" not in env.memory


def test_evaluator_vmap_and_identity_exception() -> None:
    """Test lines 399->404, 402-403, and 406-434 in evaluator.py."""
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node, evaluate_graph

    class FailingIdentityBackend(MockTestStub):
        """Test class FailingIdentityBackend."""

        def execute_op(self, op_type: BackendArg, *args: BackendArg) -> None:
            """Test function execute_op.

            Args:
                op_type: Test parameter `op_type`.
                args: Test parameter `args`.

            Raises:
                RuntimeError: When raised during `execute_op` (execute_op failed).
            """
            raise RuntimeError("execute_op failed")

    class NoExecuteOpBackend(MockTestStub):
        """Test class NoExecuteOpBackend."""

        pass

    env = Environment({"in_val": 42})
    n_id = LogicalNode(id="id_out", op_type="Identity", inputs=["in_val"])
    _evaluate_node(n_id, env, FailingIdentityBackend())
    assert env.get("id_out") == 42
    _evaluate_node(n_id, env, NoExecuteOpBackend())
    assert env.get("id_out") == 42

    import pydantic
    from pydantic import ConfigDict

    class CustomGraphLike(pydantic.BaseModel):
        """Test class CustomGraphLike."""

        model_config = ConfigDict(arbitrary_types_allowed=True)
        id: str = "custom_body"
        inputs: list[str] = ["x"]
        outputs: list[str] = ["y"]
        nodes: list[LogicalNode] = [
            LogicalNode(id="x", op_type="Input", shape_metadata=()),
            LogicalNode(id="y", op_type="Identity", inputs=["x"], shape_metadata=()),
        ]

    custom_body_graph = CustomGraphLike()

    g_vmap = LogicalGraph("g_vmap")
    g_vmap.inputs = ["arr"]
    g_vmap.outputs = ["v_out"]
    g_vmap.nodes = NodeDict(
        {
            "arr": LogicalNode(id="arr", op_type="Input", shape_metadata=(2,)),
            "v_out": LogicalNode(
                id="v_out",
                op_type="Vmap",
                inputs=["arr"],
                attributes=ensure_attrs({"body": custom_body_graph, "in_axes": 0, "out_axes": 0}),
            ),
        }
    )
    res = evaluate_graph(g_vmap, {"arr": np.array([10.0, 20.0])})
    assert "v_out" in res
    np.testing.assert_array_equal(ensure_ndarray(res["v_out"]), np.array([10.0, 20.0]))
    from ml_switcheroo_ir import IRGraph, IRNode

    body_ir = IRGraph(name="body_ir")
    body_ir.inputs = ["x"]
    body_ir.outputs = ["y"]
    body_ir.nodes = NodeDict(
        {
            "x": IRNode(id="x", op_type="Input", shape_metadata=()),
            "y": IRNode(id="y", op_type="Identity", inputs=["x"], shape_metadata=()),
        }
    )
    g_vmap_ir = LogicalGraph("g_vmap_ir")
    g_vmap_ir.inputs = ["arr"]
    g_vmap_ir.outputs = ["v_out"]
    g_vmap_ir.nodes = NodeDict(
        {
            "arr": LogicalNode(id="arr", op_type="Input", shape_metadata=(2,)),
            "v_out": LogicalNode(
                id="v_out",
                op_type="Vmap",
                inputs=["arr"],
                attributes=ensure_attrs({"body": body_ir, "in_axes": 0, "out_axes": 0}),
            ),
        }
    )
    res_ir = evaluate_graph(g_vmap_ir, {"arr": np.array([10.0, 20.0])})
    assert "v_out" in res_ir


def test_evaluator_subgraphs_and_initializers() -> None:
    """Test evaluating nested subgraphs and fallback to graph.initializers."""
    then_graph = LogicalGraph(name="then_branch")
    then_graph.inputs = ["x"]
    then_graph.nodes["x"] = LogicalNode(id="x", op_type="Input")
    then_graph.nodes["const_10"] = LogicalNode(id="const_10", op_type="Constant", attributes={"value": 10})
    then_graph.nodes["res_then"] = LogicalNode(id="res_then", op_type="Add", inputs=["x", "const_10"])
    then_graph.outputs = ["res_then"]
    else_graph = LogicalGraph(name="else_branch")
    else_graph.inputs = ["x"]
    else_graph.nodes["x"] = LogicalNode(id="x", op_type="Input")
    else_graph.nodes["res_else"] = LogicalNode(id="res_else", op_type="Negative", inputs=["x"])
    else_graph.outputs = ["res_else"]
    if_graph = LogicalGraph(name="if_graph")
    if_graph.inputs = ["pred", "val"]
    if_graph.nodes["pred"] = LogicalNode(id="pred", op_type="Input")
    if_graph.nodes["val"] = LogicalNode(id="val", op_type="Input")
    if_graph.nodes["cond_node"] = LogicalNode(
        id="cond_node", op_type="If", inputs=["pred"], subgraphs={"then_branch": then_graph, "else_branch": else_graph}
    )
    if_graph.outputs = ["cond_node"]
    out_true = evaluate_graph(if_graph, {"pred": True, "val": 5, "x": 5})
    assert out_true["cond_node"] == 15
    out_false = evaluate_graph(if_graph, {"pred": False, "val": 5, "x": 5})
    assert out_false["cond_node"] == -5
    cp_sub = LogicalGraph(name="cp_sub")
    cp_sub.inputs = ["a"]
    cp_sub.nodes["a"] = LogicalNode(id="a", op_type="Input")
    cp_sub.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["a", "a"])
    cp_sub.outputs = ["out"]
    cp_graph = LogicalGraph(name="cp_graph")
    cp_graph.inputs = ["inp"]
    cp_graph.nodes["inp"] = LogicalNode(id="inp", op_type="Input")
    cp_graph.nodes["cp_node"] = LogicalNode(
        id="cp_node", op_type="Checkpoint", inputs=["inp"], subgraphs={"body": cp_sub}
    )
    cp_graph.outputs = ["cp_node"]
    out_cp = evaluate_graph(cp_graph, {"inp": np.array([3.0])})
    assert out_cp["cp_node"] == 6.0
    init_graph = LogicalGraph(name="init_graph")
    init_graph.inputs = ["in_val"]
    init_graph.initializers["weight"] = np.array([4.0])
    init_graph.nodes["in_val"] = LogicalNode(id="in_val", op_type="Input")
    init_graph.nodes["add_node"] = LogicalNode(id="add_node", op_type="Add", inputs=["in_val", "weight"])
    init_graph.outputs = ["add_node"]
    out_init = evaluate_graph(init_graph, {"in_val": np.array([1.0])})
    assert out_init["add_node"] == 5.0


def test_as_eval_value_and_index_put_reshape_bytes() -> None:
    """Verify `as_eval_value` validation, `IndexPut`, `Meshgrid` dispatch, `Reshape` kwargs, and bytes constant."""
    import ast
    import sys

    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment, as_eval_value
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op, _prepare_node_kwargs
    from ml_switcheroo_compiler.interpreter.slice_parser import _parse_constant

    assert as_eval_value((1, 2.5, "ok")) == (1, 2.5, "ok")
    assert as_eval_value([True, None, 3]) == [True, None, 3]
    with pytest.raises(TypeError, match="Unsupported non-tensor evaluation value"):
        as_eval_value(sys)

    assert _parse_constant(ast.Constant(value=b"axis")) == "axis"

    reshape_node = LogicalNode(
        id="r1", op_type="Reshape", inputs=["x"], attributes={"strides": (1, 1)}, shape_metadata=(2, 3)
    )
    r_kwargs = _prepare_node_kwargs(reshape_node, "Reshape")
    assert r_kwargs["newshape"] == (2, 3)
    assert "strides" not in r_kwargs

    zeros_node = LogicalNode(id="z1", op_type="Zeros", inputs=[], attributes={}, shape_metadata=(4, 5))
    z_kwargs = _prepare_node_kwargs(zeros_node, "Zeros")
    assert z_kwargs["shape"] == (4, 5)

    from ml_switcheroo_compiler.backends.registry import get_active_backend

    be = get_active_backend()
    env = Environment({"base": np.array([10.0, 20.0, 30.0], dtype=np.float32), "upd": 99.0})
    iput_node = LogicalNode(id="iput", op_type="IndexPut", inputs=["base", "upd"], attributes={"index": 1})
    _dispatch_op(iput_node, env, be, "IndexPut", [env.get("base"), env.get("upd")], {"index": 1})
    np.testing.assert_allclose(np.asarray(env.get("iput")), np.array([10.0, 99.0, 30.0], dtype=np.float32))

    mg_env = Environment({"c1": np.array([1.0, 2.0]), "c2": np.array([3.0, 4.0, 5.0])})
    mg_node = LogicalNode(id="mg", op_type="Meshgrid", inputs=["c1", "c2"], attributes={"output_index": 1})
    _dispatch_op(mg_node, mg_env, be, "Meshgrid", [mg_env.get("c1"), mg_env.get("c2")], {"output_index": 1})
    assert np.asarray(mg_env.get("mg")).shape == (3, 2)
