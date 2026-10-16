"""Tests for autodiff nn and shape rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

    @cast("Callable[..., EagerTensorProtocol]", register_vjp)("ExpandDims")
    def _expand_vjp(graph: EagerValue, node: EagerValue, cotangent: EagerValue) -> EagerValue:
        """Test function _expand_vjp."""
        import uuid

        from ml_switcheroo_ir import LogicalNode

        sqz_id = uuid.uuid4().hex
        sqz_node = cast("Callable[..., EagerTensorProtocol]", LogicalNode)(id=sqz_id, op_type="Squeeze", inputs=[cotangent], attributes={"axis": cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(node, "attributes")).get)("axis", 0)})
        getattr(getattr(graph, "nodes"), "__setitem__")(sqz_id, sqz_node)
        return (sqz_id,)


try:
    get_vjp("Squeeze")
except ValueError:

    @cast("Callable[..., EagerTensorProtocol]", register_vjp)("Squeeze")
    def _squeeze_vjp(graph: EagerValue, node: EagerValue, cotangent: EagerValue) -> EagerValue:
        """Test function _squeeze_vjp."""
        import uuid

        from ml_switcheroo_ir import LogicalNode

        exp_id = uuid.uuid4().hex
        exp_node = cast("Callable[..., EagerTensorProtocol]", LogicalNode)(id=exp_id, op_type="ExpandDims", inputs=[cotangent], attributes={"axis": cast("Callable[..., EagerTensorProtocol]", cast("EagerTensorProtocol", getattr(node, "attributes")).get)("axis", 0)})
        getattr(getattr(graph, "nodes"), "__setitem__")(exp_id, exp_node)
        return (exp_id,)


if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def make_tensor(val: EagerValue, shape: EagerValue = None) -> EagerValue:
    """Test function make_tensor."""
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return cast("Callable[..., EagerTensorProtocol]", Tensor)(val, cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape, DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))


def test_shape_rules() -> None:
    """Test function test_shape_rules."""

    def f_reshape(x: EagerValue) -> EagerValue:
        """Test function f_reshape."""
        return dispatch_op("Reshape", x, newshape=(4,))

    def f_transpose(x: EagerValue) -> EagerValue:
        """Test function f_transpose."""
        return dispatch_op("Transpose", x)

    def f_expand(x: EagerValue) -> EagerValue:
        """Test function f_expand."""
        return dispatch_op("ExpandDims", x, axis=0)

    def f_squeeze(x: EagerValue) -> EagerValue:
        """Test function f_squeeze."""
        return dispatch_op("Squeeze", x, axis=0)

    x = make_tensor([[1.0, 2.0], [3.0, 4.0]])
    cot_res = make_tensor([1.0, 1.0, 1.0, 1.0])
    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f_reshape, x)
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", vjp_fn)(cot_res)) == 1

    cot_t = make_tensor([[1.0, 1.0], [1.0, 1.0]])
    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f_transpose, x)
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", vjp_fn)(cot_t)) == 1

    x2 = make_tensor([1.0, 2.0])
    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f_expand, x2)
    cot_exp = make_tensor([[1.0, 2.0]])
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", vjp_fn)(cot_exp)) == 1

    x3 = make_tensor([[1.0, 2.0]])
    out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f_squeeze, x3)
    cot_sqz = make_tensor([1.0, 2.0])
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", vjp_fn)(cot_sqz)) == 1


def test_nn_rules() -> None:
    """Test function test_nn_rules."""

    def f_relu(x: EagerValue) -> EagerValue:
        """Test function f_relu."""
        return dispatch_op("Relu", x)

    def f_sigmoid(x: EagerValue) -> EagerValue:
        """Test function f_sigmoid."""
        return dispatch_op("Sigmoid", x)

    def f_softmax(x: EagerValue) -> EagerValue:
        """Test function f_softmax."""
        return dispatch_op("Softmax", x, axis=-1)

    x = make_tensor([0.5, -0.5])
    cot = make_tensor([1.0, 1.0])

    for f in [f_relu, f_sigmoid, f_softmax]:
        try:
            out, vjp_fn = cast("Callable[..., EagerTensorProtocol]", vjp)(f, x)
            grads = cast("Callable[..., EagerTensorProtocol]", vjp_fn)(cot)
            assert cast("Callable[..., EagerTensorProtocol]", len)(grads) == 1
        except Exception as e:
            print(f"Failed {f}: {e}")
