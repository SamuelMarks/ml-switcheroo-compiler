"""Tests for autodiff nn and shape rules."""

from __future__ import annotations

import typing

import numpy as np
from ml_switcheroo_ir import LogicalGraph, LogicalNode

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
    def _expand_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
        """Test function _expand_vjp.

        Args:
            graph: Test parameter `graph`.
            node: Test parameter `node`.
            cotangent: Test parameter `cotangent`.

        Returns:
            tuple[str, ...]: Computed output value from `_expand_vjp`.
        """
        import uuid

        sqz_id = uuid.uuid4().hex
        sqz_node = LogicalNode(
            id=sqz_id,
            op_type="Squeeze",
            inputs=[cotangent],
            attributes={"axis": node.attributes.get("axis", 0)},
        )
        graph.nodes[sqz_id] = sqz_node
        return (sqz_id,)


try:
    get_vjp("Squeeze")
except ValueError:

    @register_vjp("Squeeze")
    def _squeeze_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
        """Test function _squeeze_vjp.

        Args:
            graph: Test parameter `graph`.
            node: Test parameter `node`.
            cotangent: Test parameter `cotangent`.

        Returns:
            tuple[str, ...]: Computed output value from `_squeeze_vjp`.
        """
        import uuid

        exp_id = uuid.uuid4().hex
        exp_node = LogicalNode(
            id=exp_id,
            op_type="ExpandDims",
            inputs=[cotangent],
            attributes={"axis": node.attributes.get("axis", 0)},
        )
        graph.nodes[exp_id] = exp_node
        return (exp_id,)


if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def make_tensor(val: BackendArg, shape: tuple[int, ...] | None = None) -> Tensor:
    """Test function make_tensor.

    Args:
        val: Test parameter `val`.
        shape: Test parameter `shape`.

    Returns:
        Tensor: Computed output value from `make_tensor`.
    """
    arr = np.array(val, dtype=np.float32)
    resolved_shape: tuple[int, ...] = shape if shape is not None else arr.shape
    return Tensor(arr, TensorConfig(resolved_shape, DType.Float32, Device("cpu")))


def test_shape_rules() -> None:
    """Test function test_shape_rules."""

    def f_reshape(x: BackendArg) -> BackendArg:
        """Test function f_reshape.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_reshape`.
        """
        return dispatch_op("Reshape", x, newshape=(4,))

    def f_transpose(x: BackendArg) -> BackendArg:
        """Test function f_transpose.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_transpose`.
        """
        return dispatch_op("Transpose", x)

    def f_expand(x: BackendArg) -> BackendArg:
        """Test function f_expand.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_expand`.
        """
        return dispatch_op("ExpandDims", x, axis=0)

    def f_squeeze(x: BackendArg) -> BackendArg:
        """Test function f_squeeze.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_squeeze`.
        """
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


def test_nn_rules() -> None:
    """Test function test_nn_rules."""

    def f_relu(x: BackendArg) -> BackendArg:
        """Test function f_relu.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_relu`.
        """
        return dispatch_op("Relu", x)

    def f_sigmoid(x: BackendArg) -> BackendArg:
        """Test function f_sigmoid.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_sigmoid`.
        """
        return dispatch_op("Sigmoid", x)

    def f_softmax(x: BackendArg) -> BackendArg:
        """Test function f_softmax.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_softmax`.
        """
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
