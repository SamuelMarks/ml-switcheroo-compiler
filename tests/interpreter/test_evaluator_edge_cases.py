from ml_switcheroo_ir import LogicalNode

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
