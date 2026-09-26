from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor, NumpyGenerator, NumpyTypeTranslator


def test_numpy_type_translator():
    assert NumpyTypeTranslator.get_fallback_prefix() == "np"


def test_numpy_ast_visitor():
    kwargs = {"equation": "ij,jk->ik", "dimension": 1, "other": "val"}
    assert NumpyASTVisitor._format_kwargs(kwargs) == "other=val, axis=1"

    node = MagicMock(op_type="Parameter")
    node.id = "p1"
    assert NumpyASTVisitor.visit_Parameter(node, ["p1"]) == "p1 = None # Parameter"

    assert NumpyASTVisitor.visit_Return(None, []) == "return None"
    assert NumpyASTVisitor.visit_Return(None, ["v1"]) == "return v1"
    assert NumpyASTVisitor.visit_Return(None, ["v1", "v2"]) == "return v1, v2"

    node = MagicMock(op_type="Sum")
    assert NumpyASTVisitor.generic_visit(node, ["a"], dimension=1) == "np.sum(a, axis=1)"
    assert NumpyASTVisitor.generic_visit(node, []) == "np.sum()"
    assert NumpyASTVisitor.generic_visit(node, ["a"], keepdims=True) == "np.sum(a, keepdims=True)"


def test_numpy_generator():
    gen = NumpyGenerator(MagicMock(nodes=[]))
    assert gen.get_fallback_prefix() == "np"
    assert gen._get_backend_prefix() == "np"

    node = MagicMock(op_type="PowerIteration")
    node.attributes = {"num_iters": 5}

    assert gen.get_numpy_rng() is not None

    with patch("numpy.load", return_value="load"):
        assert gen.load("file.npy") == "load"

    with patch("numpy.save"):
        gen.save("file.npy", "arr")

    with patch("numpy.savez"):
        gen.savez("file.npz", "arr")

    with patch("numpy.savez_compressed"):
        gen.savez_compressed("file.npz", "arr")


def test_numpy_generator_helpers():
    gen = NumpyGenerator(MagicMock(nodes=[]))
    assert isinstance(gen.get_helper_functions(), list)
    assert NumpyGenerator.get_numpy_rng() is not None
    import numpy as np

    res = NumpyGenerator.execute_op("Add", np.array([1]), np.array([2]))
    assert res is not None


def test_numpy_generator_aot_and_fallback_prefix():
    """Test get_fallback_prefix and compile_aot in NumpyGenerator."""
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # 1. get_fallback_prefix and visit_PowerIteration (lines 198-200)
    gen = NumpyGenerator(LogicalGraph("g"))
    assert gen.get_fallback_prefix() == "np"

    n_pwr = LogicalNode(id="pwr", op_type="PowerIteration", attributes={"num_iters": 5})
    assert gen.visit_PowerIteration(n_pwr, ["v0"]) == "np_power_iteration(v0, 5, None)"
    assert gen.visit_PowerIteration(n_pwr, ["v0", "v1"]) == "np_power_iteration(v0, 5, v1)"

    # 2. compile_aot with single output (both Tensor and raw array)
    g_single = LogicalGraph("g_single")
    g_single.inputs = ["x"]
    g_single.outputs = ["y"]
    g_single.nodes = {
        "x": LogicalNode(id="x", op_type="Input", shape_metadata=()),
        "y": LogicalNode(id="y", op_type="Identity", inputs=["x"], shape_metadata=()),
    }

    # Test passing fewer args than input nodes (branch 237->236)
    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"y": np.array([0.0])}):
        fn_aot_mock = gen.compile_aot(g_single)
        res_no_args = fn_aot_mock()
        assert res_no_args is not None

    fn_aot = gen.compile_aot(g_single)
    # Test passing raw numpy array
    res1 = fn_aot(np.array([1.0, 2.0]))
    assert isinstance(res1, Tensor)
    np.testing.assert_array_equal(res1.data, np.array([1.0, 2.0]))

    # Test passing Tensor input
    t_in = Tensor(np.array([3.0]), TensorConfig((1,), DType.Float32, Device("cpu")))
    res2 = fn_aot(t_in)
    assert isinstance(res2, Tensor)
    np.testing.assert_array_equal(res2.data, np.array([3.0]))

    # Test returning Tensor directly
    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"y": t_in}):
        fn_aot_tensor = gen.compile_aot(g_single)
        res_t = fn_aot_tensor(t_in)
        assert res_t is t_in

    # 3. compile_aot with multiple outputs
    g_multi = LogicalGraph("g_multi")
    g_multi.inputs = ["x"]
    g_multi.outputs = ["y1", "y2"]
    g_multi.nodes = {
        "x": LogicalNode(id="x", op_type="Input", shape_metadata=()),
        "y1": LogicalNode(id="y1", op_type="Identity", inputs=["x"], shape_metadata=()),
        "y2": LogicalNode(id="y2", op_type="Identity", inputs=["x"], shape_metadata=()),
    }
    # One Tensor, one raw array
    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"y1": t_in, "y2": np.array([4.0])}):
        fn_multi = gen.compile_aot(g_multi)
        outs = fn_multi(t_in)
        assert len(outs) == 2
        assert outs[0] is t_in
        assert isinstance(outs[1], Tensor)

    # 4. compile_aot with no outputs
    g_empty = LogicalGraph("g_empty")
    g_empty.outputs = []
    fn_empty = gen.compile_aot(g_empty)
    assert fn_empty() is None
