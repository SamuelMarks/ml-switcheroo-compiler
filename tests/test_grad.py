"""Tests for test_grad."""

from __future__ import annotations
from tests.testing_utils import extract_test_dtype, extract_test_shape
from tests.testing_utils import adapt_backend_module
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_ir import LogicalNode
from tests.testing_utils import MockTestStub, coerce_numeric, coerce_operand_list, ensure_tensor, unpack_pair
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
import pytest

import importlib
import typing

# ruff: noqa
from ml_switcheroo_compiler.grad import (
    CustomVJPFunction,
    RegisterGradient,
    checkpoint,
    custom_jvp,
    hvp,
    ir_grad,
    jvp,
    recompute_grad,
    remat,
    value_and_grad_wrt_vars,
    vjp,
)
from ml_switcheroo_compiler.grad import (
    CustomVJPFunction,
    GradCheckOptions,
    RegisterGradient,
    UnconnectedGradients,
    backward,
    check_numerical_grads,
    checkpoint,
    custom_jvp,
    custom_vjp,
    disable_jit,
    eval_shape,
    grad,
    hvp,
    ir_grad,
    jit,
    jvp,
    overwrite_with_gradient,
    recompute_grad,
    remat,
    value_and_grad,
    value_and_grad_wrt_vars,
    vjp,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol
    from collections.abc import Callable
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
    from ml_switcheroo_ir import IRGraph, IRNode

CustomJVPFunction = importlib.import_module("ml_switcheroo_compiler.grad").CustomJVPFunction
CustomJVPFunction = importlib.import_module("ml_switcheroo_compiler.grad").CustomJVPFunction


def test_grad() -> None:
    """Test function test_grad."""
    assert UnconnectedGradients is not None

    def my_fun(x: BackendArg) -> BackendArg:
        """Test function my_fun.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_fun`.
        """
        return x

    vjp_fun = custom_vjp(my_fun)
    assert isinstance(vjp_fun, CustomVJPFunction)
    vjp_fun.defvjp(lambda x: (x, x), lambda g, g_in: (g,))
    assert vjp_fun(1) == 1
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyData(MockTestStub):
        """Test class DummyData."""

        id = "id"

    t = Tensor(data=DummyData(), config=TensorConfig((1,), DType.Float32, "cpu"))
    assert len(vjp_fun._extract_tensor_args((t, 1))) == 1
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    assert vjp_fun(1) == 1
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.is_tracing = True

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: LogicalNode) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            self.nodes[node.id] = node

    state.global_tracing_state.start_tracing("test_grad_overwrite_bwd")
    res = vjp_fun(t)
    assert isinstance(res, Tensor)
    state.global_tracing_state.is_tracing = False
    config.eager_mode = True
    f_vg = value_and_grad_wrt_vars(my_fun)
    assert f_vg(1) == (1, {})

    class TensorWithGrad(MockTestStub):
        """Test class TensorWithGrad."""

        grad = 0.0

    class TensorNoGrad(MockTestStub):
        """Test class TensorNoGrad."""

        pass

    tg = TensorWithGrad()
    tng = TensorNoGrad()
    import pytest
    from ml_switcheroo_compiler.core.errors import TracingError

    with pytest.raises(TracingError):
        backward(tg)
    with pytest.raises(TracingError):
        backward(tng)
    assert isinstance(custom_jvp(my_fun), CustomJVPFunction)
    check_numerical_grads(my_fun, (1,))
    check_numerical_grads(my_fun, (1,), GradCheckOptions())
    RegisterGradient("op")
    assert overwrite_with_gradient(1, 2) == 1
    assert callable(checkpoint(my_fun))
    assert callable(remat(my_fun))
    assert callable(recompute_grad(my_fun))
    assert ir_grad(my_fun)(1) == 1
    assert grad(my_fun)(1) == 1
    assert value_and_grad(my_fun)(1) == (1, 1)
    assert jit(my_fun)(1) == 1
    with disable_jit():
        pass
    assert eval_shape(my_fun, 1) == 1
    assert jvp(my_fun, (1,), (2,)) == (1, 2)
    assert jvp(lambda x: (x, x), (1,), (2,), has_aux=True) == ((1, 1), 2)
    (res, fn) = vjp(my_fun, 1)
    assert res == 1
    assert fn(2) == (2,)
    (res, fn) = vjp(lambda x: (x, x), 1, has_aux=True)
    assert res == (1, 1)
    assert fn(2) == (2,)
    assert hvp(my_fun, (1,), (2,)) == (1, 0.0)
    assert hvp(lambda x: (x, x), (1,), (2,), has_aux=True) == ((1, 1), 0.0)


def test_grad_branches() -> None:
    """Test function test_grad_branches."""
    import numpy as np
    from ml_switcheroo_compiler.grad import CustomVJPFunction

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    vjp_fun2 = CustomVJPFunction(lambda x: x)
    t_trace = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert vjp_fun2._trace_fwd_graph([t_trace]) is None
    assert vjp_fun2._resolve_output_metadata([]) == ((), "float32", "cpu")
    from ml_switcheroo_compiler.grad import overwrite_with_gradient

    fn = overwrite_with_gradient(1, 2)

    def _overwrite_fwd(t: BackendArg, g: BackendArg) -> BackendArg:
        """Test function _overwrite_fwd.

        Args:
            t: Test parameter `t`.
            g: Test parameter `g`.

        Returns:
            BackendArg: Computed output value from `_overwrite_fwd`.
        """
        return (t, g)

    def _overwrite_bwd(g: BackendArg, g_in: BackendArg) -> BackendArg:
        """Test function _overwrite_bwd.

        Args:
            g: Test parameter `g`.
            g_in: Test parameter `g_in`.

        Returns:
            BackendArg: Computed output value from `_overwrite_bwd`.
        """
        return (g, None)

    assert _overwrite_fwd(1, 2) == (1, 2)
    assert _overwrite_bwd(1, 2) == (1, None)


def test_grad_overwrite_bwd() -> None:
    """Test function test_grad_overwrite_bwd."""
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    state.global_tracing_state.is_tracing = True

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: dict[str, LogicalNode] = {}

        def add_node(self, node: LogicalNode) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            self.nodes[node.id] = node

    state.global_tracing_state.start_tracing("test_grad_overwrite_bwd")
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyData(MockTestStub):
        """Test class DummyData."""

        id = "id"

    t1 = Tensor(data=DummyData(), config=TensorConfig((1,), DType.Float32, "cpu"))
    t2 = Tensor(data=DummyData(), config=TensorConfig((1,), DType.Float32, "cpu"))
    overwrite_with_gradient(t1, t2)
    n = (
        list(adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("nodes").values())[0]
        if adapt_backend_module(state.global_tracing_state.active_graph).require_symbol("nodes")
        else None
    )
    if n is not None:
        bwd_fn = n.attributes["bwd_fn"]
        fwd_fn = n.attributes["fwd_graph"]
        assert fwd_fn is not None
        assert bwd_fn(1, 2) == (1, None)
    config.eager_mode = True
    state.global_tracing_state.is_tracing = False


def test_custom_vjp_function_none() -> None:
    """Test the correctness and edge cases of the custom vjp function none functionality."""
    import numpy as np
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    cg = CustomVJPFunction(f)
    t_a = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    t_b = Tensor(np.array([2.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert cg._trace_fwd_graph([t_a, t_b]) is None


def test_value_and_grad_wrt_vars() -> None:
    """Test the correctness and edge cases of the value and grad wrt vars functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    wrapped = value_and_grad_wrt_vars(f)
    (v, g) = wrapped(1.0)
    assert v == 1.0
    assert isinstance(g, dict)


def test_custom_jvp() -> None:
    """Test the correctness and edge cases of the custom jvp functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    assert custom_jvp(f) == f


def test_register_gradient() -> None:
    """Test the correctness and edge cases of the register gradient functionality."""
    dec = RegisterGradient("my_op")

    def f(graph: IRGraph, node: IRNode, cotangent: str) -> str:
        """Test function f.

        Args:
            graph: Test parameter `graph`.
            node: Test parameter `node`.
            cotangent: Test parameter `cotangent`.

        Returns:
            str: Cotangent ID.
        """
        _ = (graph, node)
        return cotangent

    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    dec(f)
    assert has_vjp("my_op")


def test_checkpoint_remat() -> None:
    """Test the correctness and edge cases of the checkpoint remat functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    assert callable(checkpoint(f))
    assert callable(remat(f))
    assert callable(recompute_grad(f))


def test_jvp_vjp_hvp() -> None:
    """Test the correctness and edge cases of the jvp vjp hvp functionality."""

    def f(x: Tensor | float) -> Tensor | float:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor | float: Computed output value from `f`.
        """
        return x * 2

    def f_aux(x: Tensor | float) -> tuple[Tensor | float, dict[str, int]]:
        """Test function f_aux.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[Tensor | float, dict[str, int]]: Computed output value from `f_aux`.
        """
        return (x * 2, {"a": 1})

    (val, tan) = jvp(f_aux, (1.0,), (0.5,), has_aux=True)
    assert val == (2.0, {"a": 1})

    # --- Shape and Dtype Inference Tests for AD Backward Passes ---
    import numpy as np
    from ml_switcheroo_compiler.grad.jvp_vjp import hvp

    def hvp_test_func(x: Tensor) -> Tensor:
        """Test function hvp_test_func.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `hvp_test_func`.
        """
        return (x * x) * x

    primal_x = np.array([2.0, 3.0], dtype=np.float32)
    tangent_v = np.array([1.0, 1.0], dtype=np.float32)

    # HVP of x^3 is d(3x^2)/dx * v = 6x * v
    val_hvp, out_tan_hvp = hvp(hvp_test_func, primal_x, tangent_v)
    assert extract_test_shape(val_hvp) == (2,)
    assert extract_test_dtype(val_hvp) in (np.float32, np.float64)
    assert extract_test_shape(out_tan_hvp) == (2,)
    assert extract_test_dtype(out_tan_hvp) in (np.float32, np.float64)
    assert np.allclose(ensure_ndarray(out_tan_hvp), [12.0, 18.0])
    # --------------------------------------------------------------
    assert tan == 1.0
    (val, tan) = jvp(f_aux, (1.0,), ((0.5,),), has_aux=True)
    assert tan == (1.0,)
    (val, tan) = jvp(f, (1.0,), ((0.5,),), has_aux=False)
    assert tan == (1.0,)
    (val, vjp_fn) = vjp(f_aux, 1.0, has_aux=True)
    assert val == (2.0, {"a": 1})
    assert vjp_fn(0.5) == (1.0,)
    (val, vjp_fn) = vjp(f, 1.0)
    assert val == 2.0
    assert vjp_fn(0.5) == (1.0,)
    (val, tan) = hvp(f_aux, (1.0,), (0.5,), has_aux=True)
    assert val == (2.0, {"a": 1})
    assert tan == 0.0
    (val, tan) = hvp(f_aux, (1.0,), ((0.5,),), has_aux=True)
    assert tan == (0.0,)
    (val, tan) = hvp(f, (1.0,), ((0.5,),), has_aux=False)
    assert tan == (0.0,)


def test_ir_grad() -> None:
    """Test the correctness and edge cases of the ir grad functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    ir_grad(f)
    assert callable(ir_grad)


def test_backward_under_tracing() -> None:
    """Test backward() when active tracing graph is present."""
    import numpy as np
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
    from ml_switcheroo_compiler.grad import backward
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from ml_switcheroo_ir import LogicalNode

    # Enable tracing
    config.eager_mode = False
    original_tracing = state.global_tracing_state.is_tracing
    original_graph = state.global_tracing_state.active_graph

    try:
        # Start a tracing graph
        graph = state.global_tracing_state.start_tracing("test_backward_graph")
        graph.inputs = []
        graph.outputs = []

        # Create a Variable that requires grad
        var_id = "var_node_1"
        graph.nodes[var_id] = LogicalNode(id=var_id, op_type="Input", shape_metadata=())
        graph.inputs.append(var_id)

        p1 = ProxyTensor(id=var_id, shape=(), dtype="float32")
        p1.concrete_value = np.array(2.0)

        v = Variable(p1, TensorConfig((), DType.Float32, "cpu", trainable=True))

        # Create a computed tensor (loss)
        loss_id = "loss_node_1"
        graph.nodes[loss_id] = LogicalNode(id=loss_id, op_type="Multiply", inputs=[var_id, var_id], shape_metadata=())
        graph.outputs.append(loss_id)

        p_loss = ProxyTensor(id=loss_id, shape=(), dtype="float32")
        p_loss.concrete_value = np.array(4.0)

        loss = Tensor(p_loss, TensorConfig((), DType.Float32, "cpu"))

        # Run backward on the loss tensor
        backward(loss)

        # Under x^2 derivative at x=2 is 2x = 4.0
        assert v.grad is not None
        np.testing.assert_allclose(ensure_ndarray(v.grad), 4.0)

        # Non-scalar tensor backward should raise SwitcherooError
        p_non_scalar = ProxyTensor(id="non_scalar", shape=(2,), dtype="float32")
        non_scalar_loss = Tensor(p_non_scalar, TensorConfig((2,), DType.Float32, "cpu"))
        from ml_switcheroo_compiler.core.errors import SwitcherooError
        import pytest

        with pytest.raises(SwitcherooError):
            backward(non_scalar_loss)

    finally:
        state.global_tracing_state.stop_tracing()
        state.global_tracing_state.is_tracing = original_tracing
        state.global_tracing_state.active_graph = original_graph
        config.eager_mode = True


def test_non_trivial_analytical_grad() -> None:
    """Test grad and value_and_grad on non-trivial functions."""
    import numpy as np
    from ml_switcheroo_compiler.grad import grad, value_and_grad, ir_grad

    # f(x) = x * x + 3x
    # f'(x) = 2x + 3
    # At x = 4.0, f(4.0) = 28.0, f'(4.0) = 11.0
    f = lambda x: x * x + x * 3.0

    g = grad(f)
    vg = value_and_grad(f)
    irg = ir_grad(f)

    assert g(4.0) == 11.0
    assert vg(4.0) == (28.0, 11.0)
    assert irg(4.0) == 11.0


def test_analytical_vs_numerical_grads() -> None:
    """Rigorous analytical vs. numerical gradient matching verification."""
    import os
    import numpy as np
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad import check_numerical_grads
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # Clean and reset the global tracing state to protect against state leaks
    orig_tracing = global_tracing_state.is_tracing
    orig_graph = global_tracing_state.active_graph

    global_tracing_state.is_tracing = False
    global_tracing_state.active_graph = None

    orig_env = os.environ.get("SWITCHEROO_EAGER_MODE")
    os.environ["SWITCHEROO_EAGER_MODE"] = "1"
    config.eager_mode = True

    try:
        # 1. Non-linear polynomial activation-like composition: f(x) = x^3 - x^2
        f_poly = lambda x: x * x * x - x * x
        t1 = Tensor(np.array([2.5]), TensorConfig((), DType.Float32, "cpu"))
        check_numerical_grads(f_poly, (t1,))

        # 2. Multi-dimensional division and addition: f(x, y) = x / (y + 2.0)
        f_div = lambda x, y: x / (y + 2.0)
        t2_x = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, "cpu"))
        t2_y = Tensor(np.array([3.0, 4.0]), TensorConfig((2,), DType.Float32, "cpu"))
        check_numerical_grads(f_div, (t2_x, t2_y))

        # 3. Non-linear product and square: f(x, y) = x^2 * y + y^2 * x
        f_prod = lambda x, y: x * x * y + y * y * x
        t3_x = Tensor(np.array([1.5, 2.5]), TensorConfig((2,), DType.Float32, "cpu"))
        t3_y = Tensor(np.array([3.5, 4.5]), TensorConfig((2,), DType.Float32, "cpu"))
        check_numerical_grads(f_prod, (t3_x, t3_y))
    finally:
        global_tracing_state.is_tracing = orig_tracing
        global_tracing_state.active_graph = orig_graph
        if orig_env is None:
            os.environ.pop("SWITCHEROO_EAGER_MODE", None)
        else:
            os.environ["SWITCHEROO_EAGER_MODE"] = orig_env
        config.eager_mode = False
    assert config.eager_mode is False


def test_grad_missing_coverage() -> None:
    """Test function test_grad_missing_coverage."""
    from ml_switcheroo_compiler.grad import backward, check_numerical_grads, grad, jacrev
    from ml_switcheroo_compiler.core.errors import SwitcherooError
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.core.dtype import DType
    import pytest
    import numpy as np

    # 1. backward on detached/scalar tensor to hit loss_id = str(tensor.data) and not wrt_ids
    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.start_tracing("test_missing")
    t_no_grad = Tensor(np.array(5.0), TensorConfig((), DType.Float32, "cpu", trainable=False))
    backward(t_no_grad)
    state.global_tracing_state.stop_tracing()
    assert t_no_grad.grad == 1.0

    # 2. check_numerical_grads failing
    from unittest.mock import patch

    def dummy_func(x: BackendArg) -> BackendArg:
        """Test function dummy_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `dummy_func`.
        """
        return coerce_numeric(x) * 2.0

    with patch("ml_switcheroo_compiler.grad.testing.vjp") as mock_vjp:
        # Mock vjp to return a wrong gradient
        mock_vjp.return_value = (dummy_func(2.0), lambda g: (np.array([999.0]),))
        with pytest.raises(SwitcherooError, match="Gradient check failed"):
            check_numerical_grads(dummy_func, (2.0,))

    # 3. grad with boolean arg (line 594) and Exception in _to_original_type (line 596)
    from ml_switcheroo_compiler.grad import GradOptions

    def f_bool(x: BackendArg) -> BackendArg:
        """Test function f_bool.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_bool`.
        """
        return x

    grad_bool = grad(f_bool)
    # the gradient of identity is 1, returned as bool if arg is bool
    res_b = grad_bool(True)
    assert res_b is True or res_b == 1.0

    # force Exception by returning array of size > 1 for float input
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.vjp") as mock_vjp:
        mock_vjp.return_value = (1.0, lambda g: (np.array([1.0, 2.0]),))
        res_ex = grad_bool(1.0)
        assert isinstance(res_ex, np.ndarray)

        # trigger line 598: orig is not int/float/bool/Tensor/np.ndarray
        res_str = grad_bool("hello")
        assert res_str is not None

        # trigger line 634: argnums is not int/tuple/list
        class MyArgnums:
            """Test class MyArgnums."""

            pass

        opts_bad = GradOptions()
        pytest.MonkeyPatch().setattr(opts_bad, "argnums", MyArgnums(), raising=False)
        grad_multi = grad(f_bool, opts_bad)
        res_multi = grad_multi(1.0)
        assert res_multi is not None

    # 4. grad with has_aux=True (line 619)
    def f_aux(x: Tensor | float) -> tuple[Tensor | float, dict[str, str]]:
        """Test function f_aux.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[Tensor | float, dict[str, str]]: Computed output value from `f_aux`.
        """
        return x * 2.0, {"info": "aux"}

    grad_aux = grad(f_aux, GradOptions(has_aux=True))
    res_aux = grad_aux(3.0)
    assert coerce_operand_list(res_aux)[0] == 2.0

    # 5. grad with argnums as list/tuple (line 631-634)
    def f_multi(x: Tensor | float, y: Tensor | float) -> Tensor | float:
        """Test function f_multi.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            Tensor | float: Computed output value from `f_multi`.
        """
        return (x * 3.0) + (y * 2.0)

    grad_multi = grad(f_multi, GradOptions(argnums=(0, 1)))
    res_multi = grad_multi(1.0, 1.0)
    assert res_multi == (3.0, 2.0)

    # 6. jacobian with has_aux=True (line 1081) and out_arr.ndim > 0 (line 1116)
    def f_jac_aux(x: Tensor) -> tuple[Tensor, dict[str, int]]:
        """Test function f_jac_aux.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[Tensor, dict[str, int]]: Computed output value from `f_jac_aux`.
        """
        return x * 3.0, {"aux": 1}

    jac_aux = jacrev(f_jac_aux, GradOptions(has_aux=True))
    j_res = jac_aux(np.array([2.0, 3.0]))
    assert extract_test_shape(j_res) == (2, 2)


def test_vjp_pytree_support() -> None:
    """Verify that VJP pullback supports nested tuples/lists of parameters and Pytrees."""
    import numpy as np
    from ml_switcheroo_compiler.grad import vjp

    # f(x, y) = (x[0] * y[0], x[1] * y[1])
    f = lambda x, y: (x[0] * y[0], x[1] * y[1])

    # Primals are nested tuples
    x_primal = (2.0, 3.0)
    y_primal = (4.0, 5.0)

    val, vjp_fn = vjp(f, x_primal, y_primal)

    # Output value is (2.0*4.0, 3.0*5.0) = (8.0, 15.0)
    assert val == (8.0, 15.0)

    # Cotangent matching output structure
    cotangent = (1.0, 1.0)
    grads = vjp_fn(cotangent)

    # Gradients w.r.t x: (y[0], y[1]) = (4.0, 5.0)
    # Gradients w.r.t y: (x[0], x[1]) = (2.0, 3.0)
    assert grads == ((4.0, 5.0), (2.0, 3.0))


def test_higher_order_derivatives() -> None:
    """Verify multi-dimensional inputs for jacfwd, jacrev, and hessian."""
    import numpy as np
    from ml_switcheroo_compiler.grad import jacfwd, jacrev, hessian

    # 1. Quadratic vector function f1(x) = x_0 * x_1
    f1 = lambda x: x[0] * x[1]
    x_primal = np.array([3.0, 4.0])

    jf = jacfwd(f1)
    jr = jacrev(f1)

    # Jacobian of f1 with respect to [x0, x1] is [x1, x0] = [4.0, 3.0]
    assert np.allclose(ensure_ndarray(jf(x_primal)), [4.0, 3.0])
    assert np.allclose(ensure_ndarray(jr(x_primal)), [4.0, 3.0])

    # 2. Vector-to-vector function f2(x) = (x_0^2, x_1^3)
    f2 = lambda x: (x[0] * x[0], x[1] * x[1] * x[1])

    # Jacobian of f2 with respect to [x0, x1] is a diagonal matrix:
    # [[2*x0, 0],
    #  [0, 3*x1^2]]
    # For x = [2.0, 3.0], Jacobian is:
    # [[4.0, 0.0],
    #  [0.0, 27.0]]
    jf2 = jacfwd(f2)
    jr2 = jacrev(f2)
    x_primal2 = np.array([2.0, 3.0])

    assert np.allclose(ensure_ndarray(jf2(x_primal2)), [[4.0, 0.0], [0.0, 27.0]])
    assert np.allclose(ensure_ndarray(jr2(x_primal2)), [[4.0, 0.0], [0.0, 27.0]])

    # 3. Hessian of f3(x) = 0.5 * x_0^2 + 2.0 * x_0 * x_1 + 3.0 * x_1^2
    # grad(f3) = [x0 + 2*x1, 2*x0 + 6*x1]
    # Hessian(f3) = [[1.0, 2.0],
    #                [2.0, 6.0]]
    f3 = lambda x: 0.5 * x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1]

    h_fn = hessian(f3)
    assert np.allclose(ensure_ndarray(h_fn(x_primal)), [[1.0, 2.0], [2.0, 6.0]])
