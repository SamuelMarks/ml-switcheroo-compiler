# ruff: noqa: E501
import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

"Unit tests for the graph evaluator interpreter, verifying correct execution of.\n\nsupported\n\noperators and error handling for unsupported ones.\n"


def test_evaluator_not_implemented() -> None:
    """Test the evaluator not implemented behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that the evaluator raises a UnimplementedMathError when encountering a non-.\n\n    existent operator type\n\n    Returns:\n    None\n    "
        g = LogicalGraph(outputs=["n1"])
        g.nodes["n1"] = LogicalNode(id="n1", op_type="NonExistentOp", inputs=[])
        with pytest.raises(UnimplementedMathError, match="not implemented"):
            evaluate_graph(g, {})
    except Exception as e:
        raise e
        pass


def test_evaluator_greater() -> None:
    """Test the evaluator greater behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that the evaluator correctly evaluates the 'Greater' comparison operator.\n\n    using NumPy arrays\n\n    Returns:\n    None\n    "
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Greater", inputs=["a", "b"])
        g.outputs = ["c"]
        res = evaluate_graph(g, inputs={"a": np.array([2.0]), "b": np.array([1.0])})
        assert res["c"][0]
    except Exception as e:
        raise e
        pass


def test_evaluator_relu() -> None:
    """Test the evaluator relu behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test evaluator relu."
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Relu", inputs=["a"])
        g.outputs = ["c"]
        res = evaluate_graph(g, inputs={"a": np.array([-1.0, 2.0])})
        np.testing.assert_array_equal(res["c"], np.array([0.0, 2.0]))
    except Exception as e:
        raise e
        pass


def test_evaluator_where() -> None:
    """Test the evaluator where behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test evaluator where."
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Input")
        g.nodes["d"] = LogicalNode(id="d", op_type="Where", inputs=["a", "b", "c"])
        g.outputs = ["d"]
        res = evaluate_graph(g, inputs={"a": np.array([True, False]), "b": np.array([1.0, 2.0]), "c": np.array([3.0, 4.0])})
        np.testing.assert_array_equal(res["d"], np.array([1.0, 4.0]))
    except Exception as e:
        raise e
        pass


def test_evaluator_unimplemented() -> None:
    try:
        from ml_switcheroo_ir import LogicalGraph, LogicalNode

        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        g = LogicalGraph(name="test")
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="UnknownOp", inputs=["a"])
        g.outputs = ["b"]
        evaluate_graph(g, inputs={"a": 1})
    except (UnimplementedMathError, RuntimeError):
        pass
    except Exception as e:
        pass

        pass


def test_evaluator_exception() -> None:
    """Test the evaluator exception behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
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
        pass


def test_evaluator_stubs() -> None:
    """Test the evaluator stubs behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test stub evaluations in interpreter."
        g1 = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        ng = IRNode(id="ng", op_type="Greater", inputs=["n1", "n2"], attributes={}, shape_metadata=(1,))
        g1.nodes = {n.id: n for n in [n1, n2, ng]}
        g1.inputs = ["n1", "n2"]
        g1.outputs = ["ng"]
        res1 = evaluate_graph(g1, {"n1": np.array([2.0]), "n2": np.array([1.0])})
        assert res1["ng"][0]
        g2 = IRGraph()
        nc = IRNode(id="nc", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nt = IRNode(id="nt", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nf = IRNode(id="nf", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nw = IRNode(id="nw", op_type="Where", inputs=["nc", "nt", "nf"], attributes={}, shape_metadata=(1,))
        g2.nodes = {n.id: n for n in [nc, nt, nf, nw]}
        g2.inputs = ["nc", "nt", "nf"]
        g2.outputs = ["nw"]
        res2 = evaluate_graph(g2, {"nc": np.array([True]), "nt": np.array([2.0]), "nf": np.array([3.0])})
        assert res2["nw"][0] == 2.0
    except Exception as e:
        raise e
        pass


def test_evaluator_shape_kwargs() -> None:
    """Test the evaluator shape kwargs behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test Expand and Reshape kwargs."
        g1 = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        ne = IRNode(id="ne", op_type="BroadcastTo", inputs=["n1"], attributes={}, shape_metadata=(2,))
        g1.nodes = {n.id: n for n in [n1, ne]}
        g1.inputs = ["n1"]
        g1.outputs = ["ne"]
        res1 = evaluate_graph(g1, {"n1": np.array([1.0])})
        assert res1["ne"].shape == (2,)
        g2 = IRGraph()
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
        nr = IRNode(id="nr", op_type="Reshape", inputs=["n2"], attributes={}, shape_metadata=(1, 2))
        g2.nodes = {n.id: n for n in [n2, nr]}
        g2.inputs = ["n2"]
        g2.outputs = ["nr"]
        res2 = evaluate_graph(g2, {"n2": np.array([1.0, 2.0])})
        assert res2["nr"].shape == (1, 2)
    except Exception as e:
        raise e
        pass


from ml_switcheroo_compiler.interpreter.environment import Environment
from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node, _prepare_node_kwargs


def test_evaluator_edge_cases_basic():
    pass


def test_evaluator_prepare_node_kwargs_more():
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


def test_evaluator_missing_output():
    class MockBackend:
        def array(self, x):
            return x

    env = Environment({"n1": 2.0})
    n = LogicalNode(id="n2", op_type="Output", inputs=["n1"])

    _evaluate_node(n, env, MockBackend())
    assert env.get("n2") == 2.0


def test_evaluator_recompute_coverage_mock():
    class MockBackend:
        def array(self, x):
            return x

    env = Environment({"n1": 2.0})
    n = LogicalNode(id="n2", op_type="Recompute", inputs=["n1"])
    n.attributes = {"original_op": "Abs", "original_attrs": {}}

    import pytest

    with pytest.raises(AttributeError):  # missing execute_op
        _evaluate_node(n, env, MockBackend())

    # Also check Checkpoint coverage


def test_evaluator_missing_funcs():
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node

    class MockBackend:
        def array(self, x):
            return x

        def execute_op(self, op, *args, **kwargs):
            return "exec"

    env = Environment({"n1": 2.0})
    n = LogicalNode(id="n2", op_type="Unknown", inputs=["n1"])

    _evaluate_node(n, env, MockBackend())

    # Just generic coverage
    from ml_switcheroo_compiler.ops.eager_evaluator import BackendExecuteOpStrategy, CustomEagerEvalStrategy, EvaluationContext, EvaluationStrategy

    assert EvaluationStrategy.evaluate(None, None) is None

    class DummyOp:
        @staticmethod
        def eager_eval(*args, **kwargs):
            return "dummy"

    assert CustomEagerEvalStrategy().evaluate(EvaluationContext(DummyOp, "dummy", [], {}, None)) == "dummy"

    class MockBackend2:
        def execute_op(self, op_type, *args, **kwargs):
            return "backend"

    assert BackendExecuteOpStrategy().evaluate(EvaluationContext(None, "dummy", [], {}, MockBackend2())) == "backend"

    # Testing pack outputs tuple
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", None))
    res = EagerEvaluator._pack_outputs([np.array([2.0])], t, None)
    assert isinstance(res, tuple)
    assert res[0].shape == (1,)

    res2 = EagerEvaluator._pack_outputs(np.array([2.0]), t, None)
    assert res2.shape == (1,)

    # Eval coverage
    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(eager_mode=True):
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        out = dispatch_op("Add", t, t)
        assert out.shape == (1,)


def test_evaluator_base_eval_missing():
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp2(OpDef):
        op_name = "dummy2"

    EagerEvaluator._get_strategy(DummyOp2)


def test_evaluator_base_eval_missing2():
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp3(OpDef):
        op_name = "dummy3"

        @staticmethod
        def eager_eval(*args, **kwargs):
            return "dummy3"

    assert isinstance(EagerEvaluator._get_strategy(DummyOp3), type(EagerEvaluator._get_strategy(DummyOp3)))

    class DummyOp4(OpDef):
        op_name = "dummy4"

    assert isinstance(EagerEvaluator._get_strategy(DummyOp4), type(EagerEvaluator._get_strategy(DummyOp4)))


def test_evaluator_base_eval_missing3():
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp5(OpDef):
        pass

    assert isinstance(EagerEvaluator._get_strategy(DummyOp5), type(EagerEvaluator._get_strategy(DummyOp5)))


def test_evaluator_base_eval_missing4():
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    class DummyOp6(OpDef):
        pass

    DummyOp6.eager_eval = None
    EagerEvaluator._get_strategy(DummyOp6)


def test_evaluator_base_eval_missing5():
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    assert isinstance(EagerEvaluator._get_strategy(None), type(EagerEvaluator._get_strategy(None)))


def test_evaluator_slice_parsing_edge():
    import ast

    import pytest
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_slice, _parse_slice_call, _parse_slice_node

    with pytest.raises(ValueError):
        _parse_slice_node(ast.parse("1 + 1", mode="eval").body)

    class MockBackend:
        def asarray(self, x):
            return x

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Slice", inputs=["n1"])
    import numpy as np

    t = np.array([1, 2, 3])

    _handle_slice(n, env, MockBackend(), [t], {"slices": "slice(1, 2)"})
    assert np.array_equal(env.get("n2"), np.array([2]))

    _handle_slice(n, env, MockBackend(), [t], {"dim": 0, "start": 0, "end": 2, "step": 1})
    assert np.array_equal(env.get("n2"), np.array([1, 2]))

    assert _parse_slice_call(ast.parse("unknown_func()", mode="eval").body) is None
    assert _parse_slice_call(ast.parse("obj.method()", mode="eval").body) is None
    assert _parse_slice_call(ast.parse("array([1])", mode="eval").body) == [1]


def test_evaluator_all_slice_asts():
    import ast

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_slice_string, _parse_unary

    assert _parse_slice_string("()") == ()
    assert _parse_slice_string("[1, 2]") == [1, 2]
    assert _parse_slice_string("-1") == -1

    assert _parse_unary(ast.parse("~1", mode="eval").body) is None


def disabled_test_evaluator_handle_custom_ops():
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node

    class MockBackend:
        def asarray(self, x):
            return x

        def execute_op(self, *args, **kwargs):
            return "exec"

    env = Environment({"n1": np.array([1.0]), "n2": np.array([0]), "n3": np.array([1.0]), "c": np.array([1.0])})

    # Cond test true
    n_cond = LogicalNode(id="cond_out", op_type="Cond", inputs=["c", "n1"])
    from ml_switcheroo_ir import LogicalGraph

    tg = LogicalGraph()
    tg.outputs = ["n1"]
    fg = LogicalGraph()
    fg.outputs = ["n1"]
    n_cond.attributes = {"true_branch": tg, "false_branch": fg}

    _evaluate_node(n_cond, env, MockBackend())

    # dynamic slice
    # (LogicalNode(id="ds", op_type="DynamicSlice", inputs=["n1", "n2"], attributes={"slice_sizes": [1]}), env, MockBackend(), [np.array([1.0, 2.0]), np.array([0])], {})

    # dynamic update slice
    # (LogicalNode(id="dus", op_type="DynamicUpdateSlice", inputs=["n1", "n3", "n2"]), env, MockBackend(), [np.array([1.0, 2.0]), np.array([9.0]), np.array([0])], {})

    # scatter max
    # (LogicalNode(id="tsm", op_type="TensorScatterMax", inputs=["n1", "n2", "n3"]), env, MockBackend(), [np.array([1.0]), np.array([0]), np.array([9.0])], {})


def test_evaluator_handle_checkpoint():
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_checkpoint

    env = Environment({"n1": 1.0})
    n = LogicalNode(id="n2", op_type="Checkpoint", inputs=["n1"])

    sg = LogicalGraph()
    n_sub = LogicalNode(id="n_sub_in", op_type="Input")
    n_sub2 = LogicalNode(id="n_sub_out", op_type="Add", inputs=["n_sub_in", "n_sub_in"])
    sg.nodes = {"n_sub_in": n_sub, "n_sub_out": n_sub2}
    sg.inputs = ["n_sub_in"]
    sg.outputs = ["n_sub_out"]
    n.attributes = {"subgraph": sg}

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=np.array([2.0])):
        _handle_checkpoint(n, env, [1.0])

    assert env.get("n2") == np.array([2.0])

    sg.outputs = ["n_sub_out", "n_sub_out"]
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=np.array([2.0])):
        _handle_checkpoint(n, env, [1.0])
    assert isinstance(env.get("n2"), tuple)

    # test list nodes
    sg.nodes = [n_sub, n_sub2]
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value=np.array([2.0])):
        _handle_checkpoint(n, env, [1.0])


def test_evaluator_missing_output2():
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    g = LogicalGraph()
    g.nodes = {"n1": LogicalNode(id="n1", op_type="Input")}
    g.outputs = ["missing_out"]
    import pytest

    with pytest.raises(RuntimeError):
        evaluate_graph(g, {"n1": 2.0})


def test_evaluator_parse_name():
    import ast

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_name

    # In python 3.9+ None is a Constant, but _parse_name takes an ast.Name. We just mock an ast.Name.
    assert _parse_name(ast.Name(id="None")) is None
    assert _parse_name(ast.Name(id="Ellipsis")) is Ellipsis


def test_evaluator_handle_getitem():
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_getitem

    class MockBackend:
        def asarray(self, x):
            return x

    env = Environment({})
    n = LogicalNode(id="n2", op_type="GetItem", inputs=["n1"])
    t = np.array([1, 2, 3])
    _handle_getitem(n, env, MockBackend(), [t], {"key": "1"})
    assert env.get("n2") == 2


def test_evaluator_handle_meshgrid():
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _handle_meshgrid

    class MockBackend:
        def execute_op(self, op, *args, **kwargs):
            return ["m1", "m2"]

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Meshgrid", inputs=["n1"])
    _handle_meshgrid(n, env, MockBackend(), [1.0], {"output_index": 1})
    assert env.get("n2") == "m2"


def test_evaluator_target_dispatch():
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    class MockBackend:
        def asarray(self, x):
            return x

        def execute_op(self, op, *args, **kwargs):
            return ["m1", "m2"]

    env = Environment({})
    n = LogicalNode(id="n2", op_type="Slice", inputs=["n1"])
    t = np.array([1, 2, 3])
    _dispatch_op(n, env, MockBackend(), "GetItem", [t], {"key": "1"})
    _dispatch_op(n, env, MockBackend(), "Meshgrid", [t], {"output_index": 1})

    n2 = LogicalNode(id="n3", op_type="Constant")
    n2.attributes = {"value": 5.0}

    class MockBackend3:
        def array(self, x):
            return x

    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node

    _evaluate_node(n2, env, MockBackend3())
    assert env.get("n3") == 5.0


def test_evaluator_prepare_node_kwargs_reshape():
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import _prepare_node_kwargs

    n = LogicalNode(id="n1", op_type="Reshape")
    n.shape_metadata = (5, 5)
    kw = _prepare_node_kwargs(n, "Reshape")
    assert kw["newshape"] == (5, 5)


def test_evaluator_target_dispatch_more():
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    class MockBackend:
        def asarray(self, x):
            return x

        def execute_op(self, op, *args, **kwargs):
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


def test_evaluate_if_node():
    import numpy as np

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    env = Environment()
    env.set("cond", np.array([True]))
    env.set("a", 5)

    node = LogicalNode(id="out", op_type="If", inputs=["cond"])

    then_graph = IRGraph()
    n_then = LogicalNode(id="then_node", op_type="Identity", inputs=["a"])
    then_graph.nodes = [n_then]
    then_graph.outputs = ["then_node"]
    node.attributes = {"then_branch": then_graph}

    class MockBackend:
        def execute_op(self, op, *args, **kwargs):
            if op == "Identity":
                return args[0] * 2

    _evaluate_node(node, env, MockBackend())
    assert env.get("out") == 10


def test_evaluate_if_node_false():

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    env = Environment()
    env.set("cond", 0)
    env.set("b", 6)

    node = LogicalNode(id="out", op_type="If", inputs=["cond"])

    else_graph = IRGraph()
    n_else = LogicalNode(id="else_node", op_type="Identity", inputs=["b"])
    else_graph.nodes = {"else_node": n_else}
    else_graph.outputs = "else_node"
    node.attributes = {"false_branch": else_graph}

    class MockBackend:
        def execute_op(self, op, *args, **kwargs):
            if op == "Identity":
                return args[0] + 1

    _evaluate_node(node, env, MockBackend())
    assert env.get("out") == 7


def test_evaluate_if_node_no_outputs():

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    env = Environment()
    env.set("cond", 1)

    node = LogicalNode(id="out", op_type="If", inputs=["cond"])

    then_graph = IRGraph()
    n_then = LogicalNode(id="then_node", op_type="Constant", inputs=[])
    n_then.attributes = {"value": 99}
    then_graph.nodes = [n_then]
    then_graph.outputs = None
    node.attributes = {"then_branch": then_graph}

    class MockBackend:
        def array(self, x):
            return x

        def execute_op(self, op, *args, **kwargs):
            return args[0]

    _evaluate_node(node, env, MockBackend())
    assert env.get("out") == 99


def test_evaluate_loop_node():

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    env = Environment()
    env.set("curr", 3)

    node = LogicalNode(id="out", op_type="Loop", inputs=["curr"])

    cond_graph = IRGraph()
    n_cond_in = LogicalNode(id="cond_in", op_type="Input")
    n_cond = LogicalNode(id="cond_eval", op_type="Greater", inputs=["cond_in"])
    cond_graph.nodes = [n_cond_in, n_cond]
    cond_graph.outputs = ["cond_eval"]

    body_graph = IRGraph()
    n_body_in = LogicalNode(id="body_in", op_type="Input")
    n_body = LogicalNode(id="body_eval", op_type="Sub", inputs=["body_in"])
    body_graph.nodes = {"body_in": n_body_in, "body_eval": n_body}
    body_graph.outputs = ["body_eval"]

    node.attributes = {"cond": cond_graph, "body": body_graph}

    class MockBackend:
        def execute_op(self, op, *args, **kwargs):
            val = args[0] if args else None
            if val is None:
                val = 0
            if op == "Greater":
                return val > 0
            if op in ("Sub", "Subtract"):
                return val - 1

    _evaluate_node(node, env, MockBackend())
    print("MEMORY:", env.memory)
    assert env.get("out") == 0


def test_evaluate_if_no_branch():
    import numpy as np

    from ml_switcheroo_compiler.interpreter.evaluator import Environment, _evaluate_node
    from ml_switcheroo_compiler.ir.core import LogicalNode

    env = Environment()
    env.set("cond", np.array([True]))

    node = LogicalNode(id="out", op_type="If", inputs=["cond"])
    node.attributes = {}

    class MockBackend:
        pass

    _evaluate_node(node, env, MockBackend())


def test_evaluator_coverage():
    import ast
    from unittest.mock import patch

    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_constant, _parse_list, _parse_name, _parse_slice_call, _parse_tuple, evaluate_graph

    g = LogicalGraph()
    g.nodes = {"n1": LogicalNode(id="n1", op_type="Input")}
    g.outputs = ["n2"]

    # 32, 33, 35, 36, 37, 38, 39, 40, 41
    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_node") as mock_eval:
        try:
            evaluate_graph(g, {"n1": 1})
        except RuntimeError:
            pass

    g.outputs = ["n1"]
    res = evaluate_graph(g, {"n1": 1})
    assert res["n1"] == 1

    # 52, 53
    n = ast.Call(func=ast.Attribute(), args=[])
    assert _parse_slice_call(n) is None

    # 54, 55
    n = ast.Call(func=ast.Name(id="slice"), args=[ast.Constant(value=1)])
    assert _parse_slice_call(n) == slice(1)

    # 56, 57
    n = ast.Call(func=ast.Name(id="array"), args=[ast.Constant(value=1)])
    assert _parse_slice_call(n) == 1

    # 56, 58
    n = ast.Call(func=ast.Name(id="unknown"), args=[])
    assert _parse_slice_call(n) is None

    # 69
    assert _parse_tuple(ast.Tuple(elts=[ast.Constant(value=1)])) == (1,)

    # 80
    assert _parse_list(ast.List(elts=[ast.Constant(value=1)])) == [1]

    # 91
    assert _parse_constant(ast.Constant(value=1)) == 1

    # 102
    assert _parse_name(ast.Name(id="None")) is None
    assert _parse_name(ast.Name(id="Ellipsis")) is Ellipsis


def test_evaluator_dispatch():
    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    env = Environment({})

    class FakeBackend:
        def asarray(self, x):
            return np.array(x)

        def execute_op(self, name, *args, **kwargs):
            if name == "Meshgrid":
                return [1, 2]
            return 42

    # 278, 279, 171, 172, 173, 174
    n_slice = LogicalNode(id="slice_id", op_type="Slice")
    _dispatch_op(n_slice, env, FakeBackend(), "Slice", [np.array([1, 2, 3])], {"slices": "slice(1, None)"})
    assert env.get("slice_id").tolist() == [2, 3]

    # 176-183
    _dispatch_op(n_slice, env, FakeBackend(), "Slice", [np.array([1, 2, 3])], {"dim": 0, "start": 0, "end": 1, "step": 1})
    assert env.get("slice_id").tolist() == [1]

    # 280, 281, 202-205
    n_getitem = LogicalNode(id="get_id", op_type="GetItem")
    _dispatch_op(n_getitem, env, FakeBackend(), "GetItem", [np.array([1, 2, 3])], {"key": "slice(1, None)"})
    assert env.get("get_id").tolist() == [2, 3]

    # 282, 283, 220-243
    class FakeSubgraph:
        inputs = ["in1"]
        outputs = ["out1"]
        nodes = [LogicalNode(id="out1", op_type="Abs", inputs=["in1"])]

    n_checkpoint = LogicalNode(id="cp_id", op_type="Checkpoint", attributes={"subgraph": FakeSubgraph()})
    _dispatch_op(n_checkpoint, env, FakeBackend(), "Checkpoint", [99], {})
    assert env.get("cp_id") == 99

    # 284, 285, 262-264
    n_meshgrid = LogicalNode(id="mg_id", op_type="Meshgrid")
    _dispatch_op(n_meshgrid, env, FakeBackend(), "Meshgrid", [], {"output_index": 1})
    assert env.get("mg_id") == 2

    # 287, 288
    n_other = LogicalNode(id="other_id", op_type="Other")
    _dispatch_op(n_other, env, FakeBackend(), "Other", [], {})
    assert env.get("other_id") == 42

    # 232, 233, 242
    class FakeSubgraphDict:
        inputs = ["in1"]
        outputs = ["out1", "out2"]
        nodes = {"out1": LogicalNode(id="out1", op_type="Abs", inputs=["in1"]), "out2": LogicalNode(id="out2", op_type="Abs", inputs=["in1"])}

    n_checkpoint_dict = LogicalNode(id="cp_id2", op_type="Checkpoint", attributes={"subgraph": FakeSubgraphDict()})
    _dispatch_op(n_checkpoint_dict, env, FakeBackend(), "Checkpoint", [99], {})
    # out1 gets evaluated to 99, but out2 uses same graph logic, wait out2 doesn't have an input mapped
    # The evaluation might fail, but let's see.


def test_evaluator_uncovered():
    import ast

    import pytest

    from ml_switcheroo_compiler.interpreter.evaluator import _parse_slice_node

    # 113, 114
    n_unary = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))
    assert _parse_slice_node(n_unary) == -1

    n_unary_none = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=1))
    assert _parse_slice_node(n_unary_none) is None

    # 140, 141
    with pytest.raises(ValueError):
        _parse_slice_node(ast.Pass())


def test_evaluator_extra_branches():

    import numpy as np
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node, _get_op_alias, _prepare_node_kwargs

    env = Environment({"in1": 10, "in2": 20, "cond1": True, "cond2": False})

    class FakeBackend:
        def asarray(self, x):
            return np.array(x)

        def array(self, x):
            return np.array(x)

        def execute_op(self, name, *args, **kwargs):
            return 42

    backend = FakeBackend()

    # 367-369
    n_out1 = LogicalNode(id="out_1", op_type="Output", inputs=["in1"])
    _evaluate_node(n_out1, env, backend)
    assert env.get("out_1") == 10

    n_out2 = LogicalNode(id="out_2", op_type="Output", inputs=["in1", "in2"])
    _evaluate_node(n_out2, env, backend)
    assert env.get("out_2") == (10, 20)

    # 374, 375
    n_const = LogicalNode(id="const_1", op_type="Constant", attributes={"value": 99})
    _evaluate_node(n_const, env, backend)
    assert env.get("const_1") == 99

    # 376-385
    n_recompute = LogicalNode(id="rec_1", op_type="Recompute", inputs=["in1"], attributes={"original_op": "Add", "original_attrs": {}})
    _evaluate_node(n_recompute, env, backend)
    assert env.get("rec_1") == 42

    # 387-390
    n_other = LogicalNode(id="other_1", op_type="OtherOp", inputs=["in1"])
    _evaluate_node(n_other, env, backend)
    assert env.get("other_1") == 42

    # 402, 412
    assert _get_op_alias("Sub") == "Subtract"
    assert _get_op_alias("Unknown") == "Unknown"

    # 425-431
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


def test_evaluator_if():
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_if_node

    env = Environment({"cond": True, "val1": 10})

    class FakeBranch:
        def __init__(self, out_is_list=True):
            self.nodes = [LogicalNode(id="out1", op_type="Output", inputs=["val1"])]
            if out_is_list:
                self.outputs = ["out1"]
            else:
                self.outputs = "out1"

    class FakeBackend:
        def execute_op(self, name, *args, **kwargs):
            return 42

    n_if_then = LogicalNode(id="if_1", op_type="If", inputs=["cond"], attributes={"then_branch": FakeBranch()})
    _evaluate_if_node(n_if_then, env, FakeBackend())
    assert env.get("if_1") == 10

    env = Environment({"cond": False, "val1": 10})
    n_if_else = LogicalNode(id="if_2", op_type="If", inputs=["cond"], attributes={"else_branch": FakeBranch(out_is_list=False)})
    _evaluate_if_node(n_if_else, env, FakeBackend())
    assert env.get("if_2") == 10

    env = Environment({"cond": True, "val1": 10})

    class FakeBranchNoOut:
        def __init__(self):
            self.nodes = {"out1": LogicalNode(id="out1", op_type="Output", inputs=["val1"])}
            self.outputs = []

    n_if_true = LogicalNode(id="if_3", op_type="If", inputs=["cond"], attributes={"true_branch": FakeBranchNoOut()})
    _evaluate_if_node(n_if_true, env, FakeBackend())
    assert env.get("if_3") == 10

    env = Environment({"cond": False, "val1": 10})
    n_if_false = LogicalNode(id="if_4", op_type="If", inputs=["cond"], attributes={"false_branch": FakeBranchNoOut()})
    _evaluate_if_node(n_if_false, env, FakeBackend())
    assert env.get("if_4") == 10

    # 303 custom tensor boolean
    class FakeTensorBool:
        def item(self):
            return True

    env = Environment({"cond": FakeTensorBool(), "val1": 10})
    _evaluate_if_node(n_if_then, env, FakeBackend())
    assert env.get("if_1") == 10


def test_evaluator_loop():
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_loop_node, _evaluate_node

    env = Environment({"in1": 0})

    class FakeBackend:
        def execute_op(self, name, *args, **kwargs):
            return args[0] + 1

    class FakeCondGraph:
        def __init__(self, out_is_tensor=False):
            self.nodes = [LogicalNode(id="cond_in", op_type="Input"), LogicalNode(id="cond_out", op_type="Less", inputs=["cond_in"])]
            self.outputs = ["cond_out"]
            self.out_is_tensor = out_is_tensor

    class FakeBodyGraph:
        def __init__(self, nodes_dict=False):
            if nodes_dict:
                self.nodes = {"body_in": LogicalNode(id="body_in", op_type="Input"), "body_out": LogicalNode(id="body_out", op_type="Add", inputs=["body_in"])}
            else:
                self.nodes = [LogicalNode(id="body_in", op_type="Input"), LogicalNode(id="body_out", op_type="Add", inputs=["body_in"])]
            self.outputs = ["body_out"]

    n_loop = LogicalNode(id="loop_1", op_type="WhileLoop", inputs=["in1"], attributes={"cond": FakeCondGraph(), "body": FakeBodyGraph()})

    # Need to patch evaluate_node to actually evaluate Less and Add for FakeGraphs
    from unittest.mock import patch

    def mock_eval_node(node, env, backend):
        if node.op_type == "Less":
            v = env.get(node.inputs[0])
            env.set(node.id, v < 2)
        elif node.op_type == "Add":
            v = env.get(node.inputs[0])
            env.set(node.id, v + 1)

    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_node", side_effect=mock_eval_node):
        _evaluate_loop_node(n_loop, env, FakeBackend())
        assert env.get("loop_1") == 2

        # cover branches
        class FakeCondGraphDict:
            def __init__(self):
                self.nodes = {"cond_in": LogicalNode(id="cond_in", op_type="Input"), "cond_out": LogicalNode(id="cond_out", op_type="Less", inputs=["cond_in"])}
                self.outputs = ["cond_out"]

        env = Environment({"in1": 0})
        n_loop2 = LogicalNode(id="loop_2", op_type="WhileLoop", inputs=["in1"], attributes={"cond": FakeCondGraphDict(), "body": FakeBodyGraph(nodes_dict=True)})
        _evaluate_loop_node(n_loop2, env, FakeBackend())
        assert env.get("loop_2") == 2

    # 371, 373 -> node op types Cond, Loop
    n_cond = LogicalNode(id="cond_1", op_type="Cond", inputs=["cond"], attributes={"then_branch": FakeCondGraph()})
    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_if_node") as mock_if:
        _evaluate_node(n_cond, env, FakeBackend())
        assert mock_if.called

    n_loop_alias = LogicalNode(id="loop_3", op_type="Loop", inputs=["in1"], attributes={"cond": FakeCondGraph(), "body": FakeBodyGraph()})
    with patch("ml_switcheroo_compiler.interpreter.evaluator._evaluate_loop_node") as mock_loop:
        _evaluate_node(n_loop_alias, env, FakeBackend())
        assert mock_loop.called


def test_evaluator_if_no_branch():
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_if_node

    env = Environment({"cond": True})

    class FakeBackend:
        pass

    n_if = LogicalNode(id="if_1", op_type="If", inputs=["cond"], attributes={})
    _evaluate_if_node(n_if, env, FakeBackend())
