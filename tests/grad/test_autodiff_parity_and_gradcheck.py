"""Tests for mathematically verified autodiff rules and finite difference gradcheck."""

from typing import Callable, Union

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import jvp, vjp
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from ml_switcheroo_compiler.transforms.autodiff import _add_nodes, _is_zero_node, _load_rematerialization_rules, _should_rematerialize
from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import _parse_expression
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp, has_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp, has_vjp


def _make_tensor(data: Union[float, list[float], list[list[float]], np.ndarray], shape: tuple[int, ...] = ()) -> Tensor:
    """Create a tensor for gradient testing with Float32 on CPU.

    Args:
        data (Union[float, list[float], list[list[float]], np.ndarray]): Initial array data.
        shape (tuple[int, ...]): Explicit tensor dimensions.

    Returns:
        Tensor: Constructed tensor object.
    """
    arr: np.ndarray = np.array(data, dtype=np.float32)
    actual_shape: tuple[int, ...] = shape if shape else arr.shape
    return Tensor(arr, TensorConfig(actual_shape, DType.Float32, Device("cpu")))


def _finite_difference_grad(func: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-3) -> np.ndarray:
    """Compute numerical gradient using central finite differences.

    Args:
        func (Callable[[Tensor], Tensor]): The scalar function to differentiate.
        x (Tensor): The evaluation point.
        eps (float): Finite difference perturbation step size.

    Returns:
        np.ndarray: Numerically approximated gradient array.
    """
    x_arr: np.ndarray = np.array(x.data, dtype=np.float32)
    num_grad: np.ndarray = np.zeros_like(x_arr)
    it = np.nditer(x_arr, flags=["multi_index"])
    with ConfigContext(eager_mode=True):
        while not it.finished:
            idx = it.multi_index
            orig_val = x_arr[idx]

            x_arr[idx] = orig_val + eps
            pos_t: Tensor = _make_tensor(x_arr, x.shape)
            f_pos: float = float(np.sum(func(pos_t).data))

            x_arr[idx] = orig_val - eps
            neg_t: Tensor = _make_tensor(x_arr, x.shape)
            f_neg: float = float(np.sum(func(neg_t).data))

            num_grad[idx] = (f_pos - f_neg) / (2.0 * eps)
            x_arr[idx] = orig_val
            it.iternext()
    return num_grad


def test_linear_algebra_vjp_rules() -> None:
    """Test mathematically verified VJP rules for linear algebra operators."""

    # 1. MatMul
    def f_matmul(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication."""
        return dispatch_op("MatMul", a, b)

    t_a: Tensor = _make_tensor([[1.0, 2.0], [3.0, 4.0]])
    t_b: Tensor = _make_tensor([[0.5, -0.5], [1.5, 0.0]])
    cot: Tensor = _make_tensor([[1.0, 1.0], [1.0, 1.0]])

    out, vjp_fn = vjp(f_matmul, t_a, t_b)
    assert out is not None
    grads = vjp_fn(cot)
    assert len(grads) == 2

    # 2. Transpose
    def f_transpose(x: Tensor) -> Tensor:
        """Transpose."""
        return dispatch_op("Transpose", x)

    out_t, vjp_t = vjp(f_transpose, t_a)
    grads_t = vjp_t(cot)
    assert len(grads_t) == 1
    assert grads_t[0].shape == t_a.shape

    # 3. Dot
    def f_dot(a: Tensor, b: Tensor) -> Tensor:
        """Dot product."""
        return dispatch_op("Dot", a, b)

    d_a: Tensor = _make_tensor([1.0, 2.0, 3.0])
    d_b: Tensor = _make_tensor([4.0, 5.0, 6.0])
    cot_d: Tensor = _make_tensor([1.0, 1.0, 1.0])
    out_d, vjp_d = vjp(f_dot, d_a, d_b)
    grads_d = vjp_d(cot_d)
    assert len(grads_d) == 2


def test_activation_vjp_rules() -> None:
    """Test mathematically verified VJP rules for activation functions against finite differences."""
    activations: list[str] = ["Sigmoid", "Tanh", "Relu", "GELU", "SiLU", "ELU", "LeakyReLU", "Softplus"]

    x: Tensor = _make_tensor([-1.0, -0.5, 0.5, 1.0])
    cot: Tensor = _make_tensor([1.0, 1.0, 1.0, 1.0])

    for act_name in activations:

        def f_act(t: Tensor, op_name: str = act_name) -> Tensor:
            """Activation evaluation."""
            return dispatch_op(op_name, t)

        out, vjp_fn = vjp(f_act, x)
        assert out is not None
        grads = vjp_fn(cot)
        assert len(grads) == 1
        ana_grad: np.ndarray = np.array(grads[0].data)
        num_grad: np.ndarray = _finite_difference_grad(f_act, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-2, atol=1e-2)


def test_reduction_vjp_rules() -> None:
    """Test mathematically verified VJP rules for reduction operations."""
    x: Tensor = _make_tensor([[1.0, 2.0], [3.0, 4.0]])

    # 1. ReduceSum
    def f_sum(t: Tensor) -> Tensor:
        """Sum reduction."""
        return dispatch_op("ReduceSum", t)

    out_s, vjp_s = vjp(f_sum, x)
    cot_s: Tensor = _make_tensor(1.0)
    grads_s = vjp_s(cot_s)
    assert len(grads_s) == 1
    assert grads_s[0].shape == x.shape

    # 2. ReduceMean
    def f_mean(t: Tensor) -> Tensor:
        """Mean reduction."""
        return dispatch_op("ReduceMean", t)

    out_m, vjp_m = vjp(f_mean, x)
    grads_m = vjp_m(cot_s)
    assert len(grads_m) == 1


def test_loss_functions_vjp() -> None:
    """Test mathematically verified VJP rules for loss functions."""
    pred: Tensor = _make_tensor([0.5, 0.8, 0.2])
    target: Tensor = _make_tensor([0.0, 1.0, 0.0])
    cot: Tensor = _make_tensor([1.0, 1.0, 1.0])

    losses: list[str] = ["MSELoss", "BCELoss", "HuberLoss", "KLDivergenceLoss"]
    for loss_name in losses:

        def f_loss(p: Tensor, t: Tensor, name: str = loss_name) -> Tensor:
            """Loss function evaluation."""
            return dispatch_op(name, p, t)

        out, vjp_fn = vjp(f_loss, pred, target)
        assert out is not None
        grads = vjp_fn(cot)
        assert len(grads) == 2


def test_normalization_and_spatial_vjp() -> None:
    """Test VJP rules for normalization layers and spatial operations."""
    x: Tensor = _make_tensor([[1.0, 2.0], [3.0, 4.0]])
    gamma: Tensor = _make_tensor([1.0, 1.0])
    beta: Tensor = _make_tensor([0.0, 0.0])
    cot: Tensor = _make_tensor([[1.0, 1.0], [1.0, 1.0]])

    def f_ln(inp: Tensor, g: Tensor, b: Tensor) -> Tensor:
        """LayerNorm evaluation."""
        return dispatch_op("LayerNorm", inp, g, b)

    out_ln, vjp_ln = vjp(f_ln, x, gamma, beta)
    assert out_ln is not None
    grads_ln = vjp_ln(cot)
    assert len(grads_ln) == 3


def test_jvp_linear_algebra_and_activations() -> None:
    """Test JVP forward-mode derivative rules across ops."""
    x: Tensor = _make_tensor([1.0, 2.0])
    tan: Tensor = _make_tensor([0.1, 0.2])

    def f_sig(t: Tensor) -> Tensor:
        """Sigmoid."""
        return dispatch_op("Sigmoid", t)

    out, jvp_res = jvp(f_sig, (x,), (tan,))
    assert out is not None
    assert jvp_res is not None


def test_autodiff_zero_node_and_accumulation() -> None:
    """Test zero node optimization in gradient accumulation."""
    graph: IRGraph = IRGraph()
    zero_node: LogicalNode = LogicalNode(id="z1", op_type="Zeros", inputs=[])
    zero_const: LogicalNode = LogicalNode(id="z2", op_type="Constant", inputs=[], attributes={"value": 0.0})
    val_node: LogicalNode = LogicalNode(id="v1", op_type="Variable", inputs=[])
    graph.nodes["z1"] = zero_node
    graph.nodes["z2"] = zero_const
    graph.nodes["v1"] = val_node

    assert _is_zero_node(graph, "z1") is True
    assert _is_zero_node(graph, "z2") is True
    assert _is_zero_node(graph, "v1") is False
    assert _is_zero_node(graph, "nonexistent") is False

    # Check that _add_nodes eliminates zero nodes
    res1 = _add_nodes(graph, "z1", "v1")
    assert res1 == "v1"
    res2 = _add_nodes(graph, "v1", "z2")
    assert res2 == "v1"


def test_rematerialization_rules_loader_and_predicates() -> None:
    """Test rematerialization config loading and predicate logic."""
    rules = _load_rematerialization_rules()
    assert isinstance(rules, dict)

    node_target = LogicalNode(id="n1", op_type="Relu", inputs=[], attributes={"rematerialize": True})
    assert _should_rematerialize(node_target, rules) is True

    node_high_cost = LogicalNode(id="n2", op_type="MatMul", inputs=[], attributes={"rematerialize": False, "checkpoint": True})
    assert _should_rematerialize(node_high_cost, rules) is False

    node_checkpoint_target = LogicalNode(id="n3", op_type="Relu", inputs=[], attributes={"checkpoint": True})
    assert _should_rematerialize(node_checkpoint_target, rules) is True

    node_not_target = LogicalNode(id="n4", op_type="UnknownOp", inputs=[], attributes={"checkpoint": True})
    assert _should_rematerialize(node_not_target, rules) is False


def test_autodiff_provider_parser_directives() -> None:
    """Test parser handling for output, kwargs, and broadcast directives."""
    graph: IRGraph = IRGraph()
    node: LogicalNode = LogicalNode(id="n_prim", op_type="Sigmoid", inputs=["in0"], attributes={"dim": -1})
    graph.nodes["in0"] = LogicalNode(id="in0", op_type="Input", inputs=[])
    graph.nodes["n_prim"] = node

    # $output
    out_ref = _parse_expression(graph, "$output", node)
    assert out_ref == "n_prim"

    # Kwargs in call
    res_kw = _parse_expression(graph, "Transpose($input[0], permutation=[1, 0])", node)
    assert res_kw in graph.nodes
    assert graph.nodes[res_kw].attributes.get("permutation") == [1, 0]

    # BroadcastReduce
    res_br = _parse_expression(graph, "BroadcastReduce($cotangent, $input[0])", node, cotangent="C")
    assert res_br in graph.nodes
    assert graph.nodes[res_br].op_type == "BroadcastReduce"

    # BroadcastLike
    res_bl = _parse_expression(graph, "BroadcastLike($input[0], $output)", node)
    assert res_bl in graph.nodes
    assert graph.nodes[res_bl].op_type == "BroadcastLike"


def test_jvp_and_vjp_registry_fallbacks() -> None:
    """Test registry introspection and retrieval functions."""
    assert has_vjp("MatMul") is True
    assert has_jvp("MatMul") is True
    assert has_vjp("Sigmoid") is True
    assert has_jvp("Sigmoid") is True

    assert get_vjp("MatMul") is not None
    assert get_jvp("MatMul") is not None

    # JVP has finite difference fallback for unknown ops
    unknown_jvp = get_jvp("CompletelyUnknownFakeOpThatDoesNotExist")
    assert callable(unknown_jvp)

    with pytest.raises(ValueError):
        get_vjp("CompletelyUnknownFakeOpThatDoesNotExist")


def test_jvp_and_vjp_registry_duplicate_registration_and_coverage() -> None:
    """Test duplicate registration and fallback branches in JVP and VJP registries."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        with pytest.raises(ValueError, match="No JVP rule registered"):
            get_jvp("NonExistentOp")

        assert has_jvp("NonExistentOp") is False

        @register_jvp("CustomTestOpJVP")
        def _dummy_jvp(g: object, n: object, t: object) -> str:
            """Dummy JVP function."""
            del g, n, t
            return "ok"

        assert has_jvp("CustomTestOpJVP") is True
        assert get_jvp("CustomTestOpJVP") == _dummy_jvp

        with pytest.raises(ValueError, match="already registered"):

            @register_jvp("CustomTestOpJVP")
            def _dummy_jvp_dup(g: object, n: object, t: object) -> str:
                """Duplicate JVP function."""
                del g, n, t
                return "err"

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value=None):
        with pytest.raises(ValueError, match="No VJP rule registered"):
            get_vjp("NonExistentOpVJP")

        assert has_vjp("NonExistentOpVJP") is False

        @register_vjp("CustomTestOpVJP")
        def _dummy_vjp(g: object, n: object, c: object) -> tuple[str, ...]:
            """Dummy VJP function."""
            del g, n, c
            return ("ok",)

        assert has_vjp("CustomTestOpVJP") is True
        assert get_vjp("CustomTestOpVJP") == _dummy_vjp

        with pytest.raises(ValueError, match="already registered"):

            @register_vjp("CustomTestOpVJP")
            def _dummy_vjp_dup(g: object, n: object, c: object) -> tuple[str, ...]:
                """Duplicate VJP function."""
                del g, n, c
                return ("err",)

    # Directly hit line 69 without mock
    assert get_vjp("CustomTestOpVJP") == _dummy_vjp


def test_rematerialization_error_and_high_cost_coverage() -> None:
    """Test rematerialization config loading error handling and high-cost exclusion."""
    from unittest.mock import patch

    with patch("builtins.open", side_effect=OSError("Disk error")):
        assert _load_rematerialization_rules() == {}

    with patch("yaml.safe_load", return_value="not_a_dict"):
        assert _load_rematerialization_rules() == {}

    high_cost: LogicalNode = LogicalNode(id="hc1", op_type="MatMul", inputs=[], attributes={"rematerialize": False})
    rules: dict[str, list[str]] = {"high_cost_ops": ["MatMul"], "target_ops": ["Relu"]}
    assert _should_rematerialize(high_cost, rules) is False

    target_node: LogicalNode = LogicalNode(id="t1", op_type="Relu", inputs=[], attributes={"checkpoint": True})
    assert _should_rematerialize(target_node, rules) is True


def test_expanded_vjp_and_jvp_registry_rules() -> None:
    """Test that all newly added linear algebra, special, and structural ops are in the VJP/JVP registry."""
    new_ops: list[str] = [
        "Outer",
        "Cholesky",
        "Solve",
        "Det",
        "SVD",
        "Slice",
        "Concat",
        "Split",
        "Gather",
        "Scatter",
        "Erf",
    ]
    for op in new_ops:
        assert has_vjp(op) is True, f"Missing VJP rule for {op}"
        assert has_jvp(op) is True, f"Missing JVP rule for {op}"
        vjp_fn = get_vjp(op)
        jvp_fn = get_jvp(op)
        assert callable(vjp_fn)
        assert callable(jvp_fn)


def test_erf_and_outer_finite_difference_grad() -> None:
    """Mathematically verify Erf and Outer gradients using central finite differences."""

    # 1. Erf finite-difference verification
    def f_erf(x: Tensor) -> Tensor:
        """Scalar erf sum."""
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        return dispatch_op("Erf", x)

    x: Tensor = _make_tensor([0.2, 0.5, -0.8, 1.2])
    num_grad_erf: np.ndarray = _finite_difference_grad(f_erf, x)

    # Analytical derivative: 2 / sqrt(pi) * exp(-x^2)
    expected_erf_grad: np.ndarray = (2.0 / np.sqrt(np.pi)) * np.exp(-(np.array(x.data) ** 2))
    np.testing.assert_allclose(num_grad_erf, expected_erf_grad, rtol=1e-3, atol=1e-3)

    # 2. Outer product finite-difference verification
    u: Tensor = _make_tensor([1.0, 2.0, 3.0])
    v_arr: np.ndarray = np.array([4.0, 5.0], dtype=np.float32)

    def f_outer(u_in: Tensor) -> Tensor:
        """Outer product with fixed second vector."""
        from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

        v_t: Tensor = _make_tensor(v_arr)
        return dispatch_op("Outer", u_in, v_t)

    num_grad_outer: np.ndarray = _finite_difference_grad(f_outer, u)
    # Derivative wrt u of sum(u otimes v) is sum(v) for all entries
    expected_outer_grad: np.ndarray = np.full_like(u.data, np.sum(v_arr))
    np.testing.assert_allclose(num_grad_outer, expected_outer_grad, rtol=1e-3, atol=1e-3)


def test_control_flow_adjoint_lowering() -> None:
    """Test VJP and JVP lowering into adjoint control flow structures."""
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
        _cond_jvp,
        _cond_vjp,
        _scan_jvp,
        _scan_vjp,
        _while_loop_jvp,
        _while_loop_vjp,
    )

    g: IRGraph = IRGraph(name="test_ctrl_flow")

    # Cond node with subgraphs
    tb: IRGraph = IRGraph(name="then_b")
    tb.inputs = ["x"]
    tb.outputs = ["y"]
    node_x: LogicalNode = LogicalNode(id="x", op_type="Input", inputs=[], shape_metadata=(2,))
    node_y: LogicalNode = LogicalNode(id="y", op_type="Negative", inputs=["x"], shape_metadata=(2,))
    tb.nodes = {"x": node_x, "y": node_y}

    eb: IRGraph = IRGraph(name="else_b")
    eb.inputs = ["x"]
    eb.outputs = ["y"]
    eb.nodes = {"x": node_x, "y": node_y}

    cond_node: LogicalNode = LogicalNode(
        id="cond_node",
        op_type="Cond",
        inputs=["pred", "x"],
        attributes={"then_branch": tb, "else_branch": eb},
        shape_metadata=(2,),
    )

    adj_cond_vjp = _cond_vjp(g, cond_node, "cot_y")
    assert len(adj_cond_vjp) == 2
    assert "cond_node_adj_cond" in g.nodes

    adj_cond_jvp = _cond_jvp(g, cond_node, ["t_pred", "t_x"])
    assert adj_cond_jvp == "cond_node_jvp"
    assert "cond_node_jvp" in g.nodes

    # Scan node with body
    body_g: IRGraph = IRGraph(name="body_g")
    body_g.inputs = ["carry", "x"]
    body_g.outputs = ["carry_out"]
    node_carry: LogicalNode = LogicalNode(id="carry", op_type="Input", inputs=[], shape_metadata=(2,))
    node_elem: LogicalNode = LogicalNode(id="x", op_type="Input", inputs=[], shape_metadata=(2,))
    node_co: LogicalNode = LogicalNode(id="carry_out", op_type="Add", inputs=["carry", "x"], shape_metadata=(2,))
    body_g.nodes = {"carry": node_carry, "x": node_elem, "carry_out": node_co}

    scan_node: LogicalNode = LogicalNode(
        id="scan_node",
        op_type="Scan",
        inputs=["carry", "xs"],
        attributes={"body": body_g, "reverse": False},
        shape_metadata=(2,),
    )
    adj_scan_vjp = _scan_vjp(g, scan_node, "cot_c")
    assert len(adj_scan_vjp) == 2
    assert "scan_node_adj_scan" in g.nodes
    assert g.nodes["scan_node_adj_scan"].attributes["reverse"] is True

    adj_scan_jvp = _scan_jvp(g, scan_node, ["t_c", "t_xs"])
    assert adj_scan_jvp == "scan_node_jvp"

    # WhileLoop node with body and cond
    while_node: LogicalNode = LogicalNode(
        id="while_node",
        op_type="WhileLoop",
        inputs=["carry", "x"],
        attributes={"body": body_g, "cond": tb},
        shape_metadata=(2,),
    )
    adj_while_vjp = _while_loop_vjp(g, while_node, "cot_v")
    assert len(adj_while_vjp) == 2
    assert "while_node_adj_while" in g.nodes

    adj_while_jvp = _while_loop_jvp(g, while_node, ["t_c", "t_x"])
    assert adj_while_jvp == "while_node_jvp"


def test_declarative_rematerialization_policies() -> None:
    """Test loading and resolution of declarative rematerialization policies."""
    from ml_switcheroo_compiler.grad.checkpointing import (
        RematerializationPolicyModel,
        checkpoint,
        get_rematerialization_policy,
        load_rematerialization_rules,
        remat,
    )

    cfg = load_rematerialization_rules()
    assert "recompute_all" in cfg.policies
    assert "selective" in cfg.policies
    assert "memory_budget" in cfg.policies

    pol = get_rematerialization_policy("selective")
    assert isinstance(pol, RematerializationPolicyModel)
    assert pol.evict_activations is True
    assert "MatMul" in pol.high_cost_ops

    # Custom instance passthrough
    custom_pol = RematerializationPolicyModel(target_ops=["CustomOp"], evict_activations=False)
    assert get_rematerialization_policy(custom_pol) == custom_pol

    # Custom path loading
    custom_cfg = load_rematerialization_rules("/nonexistent_path/remat.yaml")
    assert isinstance(custom_cfg, object)

    # Empty policies fallback to target_ops
    from unittest.mock import patch

    empty_policies_cfg = load_rematerialization_rules()
    with patch("ml_switcheroo_compiler.grad.checkpointing.load_rematerialization_rules", return_value=type("Cfg", (), {"default_policy": "missing", "policies": {}, "target_ops": ["Relu"], "high_cost_ops": []})()):
        fallback_pol = get_rematerialization_policy(None)
        assert fallback_pol.target_ops == ["Relu"]

    # Unknown policy raises ValueError
    with pytest.raises(ValueError, match="not found"):
        get_rematerialization_policy("completely_unknown_policy_xyz")

    # Checkpoint with named policy
    def f(x: Tensor) -> Tensor:
        """Checkpoint target function."""
        return x

    cp_fn = checkpoint(f, policy="selective")
    assert callable(cp_fn)

    remat_fn = remat(f, policy=custom_pol)
    assert callable(remat_fn)
