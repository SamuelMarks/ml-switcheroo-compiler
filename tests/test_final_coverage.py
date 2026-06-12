"""Test coverage."""

import pytest
import numpy as np
from ml_switcheroo_ir import LogicalGraph, LogicalNode
import ml_switcheroo.jnp as jnp
from ml_switcheroo.core.config import config
import ml_switcheroo.tracing as tracing
from typing import NoReturn


def test_manipulation_extras() -> None:
    """Test manipulation extras."""
    config.eager_mode = True
    tracing._tracer.is_tracing = False
    arr = jnp.array([1, 2])
    jnp.concatenate([arr, arr], axis=0)
    with pytest.raises(NotImplementedError):
        jnp.ravel(arr, order="F")
    jnp.swapaxes(arr, 0, 0)
    jnp.moveaxis(arr, 0, 0)
    jnp.stack([arr, arr], axis=0)
    assert jnp.shape(arr) == (2,)


def test_nn_complex_extras() -> None:
    """Test nn complex extras."""
    from ml_switcheroo.nn.complex import rnn_cell, lstm_cell

    tracing._tracer.is_tracing = True
    tracing._tracer.active_graph = LogicalGraph()
    i = jnp.zeros((10,))
    h = jnp.zeros((20,))
    c = jnp.zeros((20,))
    wih = jnp.zeros((20, 10))
    whh = jnp.zeros((20, 20))
    bih = jnp.zeros((20,))
    bhh = jnp.zeros((20,))
    rnn_cell(i._tensor, h._tensor, wih._tensor, whh._tensor, None, None)
    lstm_cell(
        i._tensor, (h._tensor, c._tensor), wih._tensor, whh._tensor, bih._tensor, None
    )
    lstm_cell(
        i._tensor, (h._tensor, c._tensor), wih._tensor, whh._tensor, None, bhh._tensor
    )


def test_ops_emitters_jvp() -> None:
    """Test emitters and jvp."""
    from ml_switcheroo.ops.binary.math import TrueDivide

    op = TrueDivide()
    assert op.emit_pytorch("x", "y") == "torch.true_divide(x, y)"
    assert op.emit_keras("x", "y") == "keras.ops.true_divide(x, y)"

    from ml_switcheroo.ops.creation.basic import Full, Arange

    op = Full()
    assert op.emit_pytorch("s", "v") == "torch.full(s, v)"
    assert op.emit_mlx("s", "v") == "mx.full(s, v)"
    assert op.emit_keras("s", "v") == "keras.ops.full(s, v)"

    op = Arange()
    assert op.vjp("dy", "x") == ()
    assert op.jvp("dx", "x") == "0"
    assert op.emit_pytorch("x") == "torch.arange(x)"
    assert op.emit_mlx("x") == "mx.arange(x)"
    assert op.emit_keras("x") == "keras.ops.arange(x)"

    from ml_switcheroo.ops.linalg.basic import Matmul, Dot, Einsum

    op = Matmul()
    assert op.infer_shape((2, 3), (3, 4)) == (2, 4)
    assert op.infer_shape((3,), (3,)) is None
    assert op.jvp("da", "db", "a", "b") == "(jnp.matmul(da, b) + jnp.matmul(a, db))"
    op = Dot()
    assert op.jvp("da", "db", "a", "b") == "(jnp.dot(da, b) + jnp.dot(a, db))"
    assert op.emit_pytorch("a", "b") == "torch.dot(a, b)"
    assert op.emit_mlx("a", "b") == "mx.dot(a, b)"
    assert op.emit_keras("a", "b") == "keras.ops.dot(a, b)"
    op = Einsum()
    assert op.jvp("da", "s", "a") == "einsum_jvp"
    assert op.emit_pytorch("s", "a") == "torch.einsum(s, a)"
    assert op.emit_mlx("s", "a") == "mx.einsum(s, a)"
    assert op.emit_keras("s", "a") == "keras.ops.einsum(s, a)"
    assert op.emit_tensorflow("s", "a") == "tf.einsum(s, a)"

    from ml_switcheroo.ops.reductions.basic import Sum

    op = Sum()
    assert "keepdims=True" in op._format_args("x", keepdims=True)
    assert op.emit_pytorch("x", axis=0) == "torch.sum(x, dim=0)"
    assert op.emit_pytorch("x", keepdims=True) == "torch.sum(x, keepdim=True)"


def test_pass_manager_extras() -> None:
    """Test pass manager extras."""
    from ml_switcheroo.transforms.pass_manager import DAGTopologicalSorter

    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["n2"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Input")
    DAGTopologicalSorter.sort(g)


def test_constant_folding_extras() -> None:
    """Test constant folding extras."""
    from ml_switcheroo.transforms.passes.constant_folding import constant_folding_pass

    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2.0, 2.0])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c1"])
    constant_folding_pass(g)
    from ml_switcheroo.transforms.passes.constant_folding import constant_folding_pass

    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2.0])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "missing"])
    constant_folding_pass(g)

    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2.0])}
    )
    g.nodes["c2"] = LogicalNode(
        id="c2", op_type="Constant", attributes={"value": np.array([3.0])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    g.outputs = ["n1"]
    constant_folding_pass(g)


def test_dce_extras() -> None:
    """Test dce extras."""
    from ml_switcheroo.transforms.passes.dce import dce_pass

    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    dce_pass(g)


def test_lift_state_extras() -> None:
    """Test lift state extras."""
    from ml_switcheroo.transforms.passes.lift_state import lift_state_pass

    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="ReadVariable")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="AssignVariable", inputs=["n1"])
    lift_state_pass(g)


def test_shape_inference_extras() -> None:
    """Test shape inference extras."""
    from ml_switcheroo.transforms.passes.shape_inference import shape_inference_pass

    g = LogicalGraph(outputs=["out"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input", shape_metadata=(2, 2))
    g.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["in"])
    g.nodes["c"] = LogicalNode(id="c", op_type="Constant", attributes={"value": [1, 2]})
    g.nodes["b"] = LogicalNode(
        id="b", op_type="BroadcastTo", inputs=["in"], shape_metadata=(1, 2, 2)
    )
    g.nodes["r"] = LogicalNode(
        id="r", op_type="Reshape", inputs=["in"], shape_metadata=(4,)
    )
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in", "in"])
    g.nodes["unk"] = LogicalNode(id="unk", op_type="UnknownOp", inputs=["in"])
    shape_inference_pass(g)

    from ml_switcheroo.ops import register_op, OpDef

    @register_op("BadOpTest")
    class BadOpTest(OpDef):
        """Docstring."""

        def infer_shape(self, *args: object, **kwargs: object) -> NoReturn:
            raise ValueError("bad shape")

        def numpy_eval(self, *args: object, **kwargs: object) -> None: ...
        def vjp(self, *args: object, **kwargs: object) -> None: ...
        def jvp(self, *args: object, **kwargs: object) -> None: ...
        def emit_jax(self, *args: object, **kwargs: object) -> None: ...
        def emit_pytorch(self, *args: object, **kwargs: object) -> None: ...
        def emit_mlx(self, *args: object, **kwargs: object) -> None: ...
        def emit_keras(self, *args: object, **kwargs: object) -> None: ...
        def emit_tensorflow(self, *args: object, **kwargs: object) -> None: ...

    g.nodes["bad"] = LogicalNode(id="bad", op_type="BadOpTest", inputs=["in"])
    from ml_switcheroo.core.errors import CompilationError

    with pytest.raises(CompilationError):
        shape_inference_pass(g)


def test_cst_transpiler_attr_not_torch() -> None:
    """Test transpiler."""
    from ml_switcheroo.backends.cst_transpiler import transpile_source

    src = "import os\nos.path.join('a', 'b')\n"
    res = transpile_source(src, target_framework="jax")
    assert "os.path.join" in res


def test_evaluator_reshape() -> None:
    """Test evaluator reshape."""
    from ml_switcheroo.interpreter.evaluator import evaluate_graph

    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(
        id="n1", op_type="Constant", attributes={"value": [1, 2, 3, 4]}
    )
    g.nodes["n2"] = LogicalNode(
        id="n2", op_type="Reshape", inputs=["n1"], shape_metadata=(2, 2)
    )
    evaluate_graph(g, {})


def test_shape_inference_output_branches() -> None:
    """Test shape inference output branches."""
    from ml_switcheroo.transforms.passes.shape_inference import shape_inference_pass

    g = LogicalGraph(outputs=["o1", "o2"])
    g.nodes["o1"] = LogicalNode(id="o1", op_type="Output", inputs=[])
    g.nodes["o2"] = LogicalNode(id="o2", op_type="Output", inputs=["missing"])
    shape_inference_pass(g)


def test_math_ops_extras() -> None:
    """Test math ops extras."""
    config.eager_mode = True
    tracing._tracer.is_tracing = False
    arr = jnp.array([1, 2])
    jnp.clip(arr, a_min=0.0, a_max=None)
    jnp.clip(arr, a_min=None, a_max=3.0)
    try:
        jnp.sum(arr, where=True)
    except Exception:
        pass
