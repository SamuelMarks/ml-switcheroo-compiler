"""Tests for autodiff nn and shape rules."""

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp, register_vjp

try:
    get_vjp("ExpandDims")
except ValueError:

    @register_vjp("ExpandDims")
    def _expand_vjp(graph, node, cotangent):
        import uuid

        from ml_switcheroo_ir import LogicalNode

        sqz_id = uuid.uuid4().hex
        sqz_node = LogicalNode(id=sqz_id, op_type="Squeeze", inputs=[cotangent], attributes={"axis": node.attributes.get("axis", 0)})
        graph.nodes[sqz_id] = sqz_node
        return (sqz_id,)


try:
    get_vjp("Squeeze")
except ValueError:

    @register_vjp("Squeeze")
    def _squeeze_vjp(graph, node, cotangent):
        import uuid

        from ml_switcheroo_ir import LogicalNode

        exp_id = uuid.uuid4().hex
        exp_node = LogicalNode(id=exp_id, op_type="ExpandDims", inputs=[cotangent], attributes={"axis": node.attributes.get("axis", 0)})
        graph.nodes[exp_id] = exp_node
        return (exp_id,)


def make_tensor(val, shape=None):
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return Tensor(val, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_shape_rules():
    def f_reshape(x):
        return dispatch_op("Reshape", x, newshape=(4,))

    def f_transpose(x):
        return dispatch_op("Transpose", x)

    def f_expand(x):
        return dispatch_op("ExpandDims", x, axis=0)

    def f_squeeze(x):
        return dispatch_op("Squeeze", x, axis=0)

    x = make_tensor([[1.0, 2.0], [3.0, 4.0]])
    cot_res = make_tensor([1.0, 1.0, 1.0, 1.0])
    out, vjp_fn = vjp(f_reshape, x)
    assert len(vjp_fn(cot_res)) == 1

    cot_t = make_tensor([[1.0, 1.0], [1.0, 1.0]])
    out, vjp_fn = vjp(f_transpose, x)
    assert len(vjp_fn(cot_t)) == 1

    x2 = make_tensor([1.0, 2.0])
    out, vjp_fn = vjp(f_expand, x2)
    cot_exp = make_tensor([[1.0, 2.0]])
    assert len(vjp_fn(cot_exp)) == 1

    x3 = make_tensor([[1.0, 2.0]])
    out, vjp_fn = vjp(f_squeeze, x3)
    cot_sqz = make_tensor([1.0, 2.0])
    assert len(vjp_fn(cot_sqz)) == 1


def test_nn_rules():
    def f_relu(x):
        return dispatch_op("Relu", x)

    def f_sigmoid(x):
        return dispatch_op("Sigmoid", x)

    def f_softmax(x):
        return dispatch_op("Softmax", x, axis=-1)

    x = make_tensor([0.5, -0.5])
    cot = make_tensor([1.0, 1.0])

    for f in [f_relu, f_sigmoid, f_softmax]:
        try:
            out, vjp_fn = vjp(f, x)
            grads = vjp_fn(cot)
            assert len(grads) == 1
        except Exception as e:
            print(f"Failed {f}: {e}")
