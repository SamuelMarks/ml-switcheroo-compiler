"""Unit tests asserting functional correctness of Pure Python backend."""

from __future__ import annotations

import math

import pytest

from ml_switcheroo_compiler.backends.pure_python.eager import execute_op
from ml_switcheroo_compiler.backends.pure_python.generator import PurePythonGenerator
from ml_switcheroo_compiler.backends.pure_python.types import PurePythonTensor
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_pure_python_tensor_basic_ops() -> None:
    """Verify basic initialization and arithmetic operations on PurePythonTensor."""
    t1: PurePythonTensor = PurePythonTensor([1.0, 2.0, 3.0])
    t2: PurePythonTensor = PurePythonTensor([4.0, 5.0, 6.0])

    assert t1.shape == (3,)
    assert t1.to_flat_list() == [1.0, 2.0, 3.0]

    added: PurePythonTensor = t1 + t2
    assert added.to_flat_list() == [5.0, 7.0, 9.0]

    added_scalar: PurePythonTensor = t1 + 10.0
    assert added_scalar.to_flat_list() == [11.0, 12.0, 13.0]

    subbed: PurePythonTensor = t2 - t1
    assert subbed.to_flat_list() == [3.0, 3.0, 3.0]

    subbed_scalar: PurePythonTensor = t2 - 1.0
    assert subbed_scalar.to_flat_list() == [3.0, 4.0, 5.0]

    multiplied: PurePythonTensor = t1 * t2
    assert multiplied.to_flat_list() == [4.0, 10.0, 18.0]

    mul_scalar: PurePythonTensor = t1 * 2.0
    assert mul_scalar.to_flat_list() == [2.0, 4.0, 6.0]

    div: PurePythonTensor = t2 / t1
    assert div.to_flat_list() == [4.0, 2.5, 2.0]

    div_scalar: PurePythonTensor = t2 / 2.0
    assert div_scalar.to_flat_list() == [2.0, 2.5, 3.0]

    neg: PurePythonTensor = -t1
    assert neg.to_flat_list() == [-1.0, -2.0, -3.0]

    assert t1.sum() == 6.0


def test_pure_python_tensor_elementwise_functions() -> None:
    """Verify unary math functions on PurePythonTensor."""
    t: PurePythonTensor = PurePythonTensor([0.0, 1.0])

    exp_t: PurePythonTensor = t.exp()
    assert math.isclose(exp_t.to_flat_list()[0], 1.0)
    assert math.isclose(exp_t.to_flat_list()[1], math.e)

    log_t: PurePythonTensor = PurePythonTensor([1.0, math.e]).log()
    assert math.isclose(log_t.to_flat_list()[0], 0.0)
    assert math.isclose(log_t.to_flat_list()[1], 1.0)

    sqrt_t: PurePythonTensor = PurePythonTensor([4.0, 9.0]).sqrt()
    assert sqrt_t.to_flat_list() == [2.0, 3.0]

    sin_t: PurePythonTensor = PurePythonTensor([0.0]).sin()
    assert math.isclose(sin_t.to_flat_list()[0], 0.0)

    cos_t: PurePythonTensor = PurePythonTensor([0.0]).cos()
    assert math.isclose(cos_t.to_flat_list()[0], 1.0)

    tanh_t: PurePythonTensor = PurePythonTensor([0.0]).tanh()
    assert math.isclose(tanh_t.to_flat_list()[0], 0.0)


def test_pure_python_scalar_and_nested_initialization() -> None:
    """Verify scalar and multi-dimensional nested array initialization."""
    scalar_float: PurePythonTensor = PurePythonTensor(3.14, dtype="float32")
    assert scalar_float.shape == ()

    scalar_int: PurePythonTensor = PurePythonTensor(5, dtype="int32")
    assert scalar_int.shape == ()

    seq_tuple: PurePythonTensor = PurePythonTensor((10.0, 20.0))
    assert seq_tuple.shape == (2,)
    assert seq_tuple.to_flat_list() == [10.0, 20.0]

    nested: PurePythonTensor = PurePythonTensor([[1.0, 2.0], [3.0, 4.0]])
    assert nested.shape == (2, 2)
    assert nested.to_flat_list() == [1.0, 2.0, 3.0, 4.0]


def test_pure_python_eager_execute_op() -> None:
    """Verify eager operator execution dispatch in Pure Python backend."""
    t1: PurePythonTensor = PurePythonTensor([2.0, 4.0])
    t2: PurePythonTensor = PurePythonTensor([1.0, 2.0])

    res_add = execute_op("Add", None, t1, t2)
    assert isinstance(res_add, PurePythonTensor)
    assert res_add.to_flat_list() == [3.0, 6.0]

    res_sub = execute_op(PurePythonGenerator, "Sub", t1, t2)
    assert isinstance(res_sub, PurePythonTensor)
    assert res_sub.to_flat_list() == [1.0, 2.0]

    res_mul = execute_op("Mul", t1, t2)
    assert isinstance(res_mul, PurePythonTensor)
    assert res_mul.to_flat_list() == [2.0, 8.0]

    res_div = execute_op("Div", t1, t2)
    assert isinstance(res_div, PurePythonTensor)
    assert res_div.to_flat_list() == [2.0, 2.0]

    res_neg = execute_op("Neg", t1)
    assert isinstance(res_neg, PurePythonTensor)
    assert res_neg.to_flat_list() == [-2.0, -4.0]

    res_exp = execute_op("Exp", t1)
    assert isinstance(res_exp, PurePythonTensor)

    res_log = execute_op("Log", t1)
    assert isinstance(res_log, PurePythonTensor)

    res_sqrt = execute_op("Sqrt", t1)
    assert isinstance(res_sqrt, PurePythonTensor)

    res_sin = execute_op("Sin", t1)
    assert isinstance(res_sin, PurePythonTensor)

    res_cos = execute_op("Cos", t1)
    assert isinstance(res_cos, PurePythonTensor)

    res_tanh = execute_op("Tanh", t1)
    assert isinstance(res_tanh, PurePythonTensor)

    res_sum = execute_op("Sum", t1)
    assert res_sum == 6.0

    res_relu = execute_op("Relu", PurePythonTensor([-1.0, 2.0]))
    assert isinstance(res_relu, PurePythonTensor)
    assert res_relu.to_flat_list() == [0.0, 2.0]

    res_sig = execute_op("Sigmoid", PurePythonTensor([0.0]))
    assert isinstance(res_sig, PurePythonTensor)
    assert math.isclose(res_sig.to_flat_list()[0], 0.5)

    res_scalars = execute_op("Add", None, 1.0, 2.0)
    assert isinstance(res_scalars, PurePythonTensor)
    assert res_scalars.to_flat_list() == [3.0]

    res_str_dispatch = execute_op("Add", "Add", t1, t2)
    assert isinstance(res_str_dispatch, PurePythonTensor)

    with pytest.raises(BackendNotSupportedError):
        execute_op("NonExistentUnsupportedOp", t1)


def test_pure_python_generator_and_aot() -> None:
    """Verify AST code generation and AOT compilation in Pure Python backend."""
    graph: IRGraph = IRGraph()
    n_in: IRNode = IRNode(id="in0", op_type="Input")
    n_param: IRNode = IRNode(id="p0", op_type="Parameter")
    n_exp: IRNode = IRNode(id="exp0", op_type="Exp", inputs=["in0"])
    n_neg: IRNode = IRNode(id="neg0", op_type="Neg", inputs=["exp0"])
    n_add: IRNode = IRNode(id="add0", op_type="Add", inputs=["exp0", "neg0"])
    n_sum: IRNode = IRNode(id="sum0", op_type="Sum", inputs=["add0"])
    n_leaf: IRNode = IRNode(id="leaf0", op_type="Exp")
    n_out: IRNode = IRNode(id="out0", op_type="Output", inputs=["sum0"])
    n_out_empty: IRNode = IRNode(id="out_empty", op_type="Output", inputs=[])

    graph.add_node(n_in)
    graph.add_node(n_param)
    graph.add_node(n_exp)
    graph.add_node(n_neg)
    graph.add_node(n_add)
    graph.add_node(n_sum)
    graph.add_node(n_leaf)
    graph.add_node(n_out)
    graph.add_node(n_out_empty)

    generator: PurePythonGenerator = PurePythonGenerator(graph)
    assert generator.get_fallback_prefix() == "math"
    assert generator.get_helper_functions() == []

    line_exp: str = generator.generic_visit(n_exp, ["in0"])
    assert "math.exp(in0)" in line_exp

    line_neg: str = generator.generic_visit(n_neg, ["exp0"])
    assert "-exp0" in line_neg

    line_add: str = generator.generic_visit(n_add, ["exp0", "neg0"])
    assert "exp0 + neg0" in line_add

    line_other: str = generator.generic_visit(IRNode(id="other", op_type="Identity"), ["in0"])
    assert "other = in0" in line_other

    line_empty: str = generator.generic_visit(IRNode(id="none", op_type="Unknown"), [])
    assert "none = None" in line_empty

    runner = generator.compile_aot(graph)
    res: PurePythonTensor = runner(PurePythonTensor([0.0]))
    assert isinstance(res, PurePythonTensor)

    # Graph without output node to cover trailing return
    no_out_graph: IRGraph = IRGraph()
    no_out_graph.add_node(IRNode(id="in1", op_type="Input"))
    no_out_graph.add_node(IRNode(id="exp1", op_type="Exp", inputs=["in1"]))
    no_out_gen: PurePythonGenerator = PurePythonGenerator(no_out_graph)
    no_out_runner = no_out_gen.compile_aot(no_out_graph)
    assert isinstance(no_out_runner(PurePythonTensor([0.0])), PurePythonTensor)

    # Graph with empty-input Output node
    empty_out_graph: IRGraph = IRGraph()
    empty_out_graph.add_node(IRNode(id="in2", op_type="Input"))
    empty_out_graph.add_node(IRNode(id="out2", op_type="Output", inputs=[]))
    empty_out_gen: PurePythonGenerator = PurePythonGenerator(empty_out_graph)
    empty_out_runner = empty_out_gen.compile_aot(empty_out_graph)
    assert isinstance(empty_out_runner(PurePythonTensor([0.0])), PurePythonTensor)


def test_pure_python_backend_registry_integration() -> None:
    """Verify BackendRegistry lazy-load retrieves PurePythonGenerator."""
    backend_cls = BackendRegistry.get("pure_python")
    assert backend_cls is PurePythonGenerator
