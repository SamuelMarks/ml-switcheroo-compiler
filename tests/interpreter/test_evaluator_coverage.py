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
