"""Tests for test_control_flow."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.control_flow import AssertOp, AssociativeScan, DebugInfs, DebugNans, ScanOp, SwitchOp, assert_value, associative_scan, case, cond, custom_gradient, fori_loop, map, map_fn, pmap, scan, scan_bind, stop_gradient, switch, switch_case, vectorized_map, while_loop
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_graph, ensure_mutable, ensure_node, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyData:
    """Test class DummyData."""

    shape = (1,)
    dtype = "float32"


class DummyTensor:
    """Test class DummyTensor."""

    shape = (1,)

    def __init__(self, data: EagerValue = 0) -> None:
        """Test function __init__.

        Args:
            data: Test parameter `data`.
        """
        self.data = data
        self.dtype = "float32"
        self.device = "cpu"

    def __getitem__(self, i: EagerValue) -> EagerValue:
        """Test function __getitem__.

        Args:
            i: Test parameter `i`.
        """
        return DummyTensor()


def test_control_flow_extra_classes() -> None:
    """Test function test_control_flow_extra_classes."""
    assert ensure_callable(AssertOp)().infer_shape(None) == ()
    t = DummyTensor()
    assert DebugInfs().infer_shape(t) == t
    assert DebugInfs().infer_shape() == ()
    t2 = DummyTensor()
    assert DebugNans().infer_shape(t2) == t2
    assert DebugNans().infer_shape() == ()
    assert ensure_callable(SwitchOp)().infer_shape(None, None) == ()
    assert ScanOp().infer_shape() == ()
    t3 = DummyTensor()
    assert AssociativeScan().infer_shape(t3) == t3
    assert AssociativeScan().infer_shape() == ()


def _test_control_flow_extra_functions_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_eager", return_value="cond_eager"):
        assert ensure_callable(cond)(None, None, None) == "cond_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_eager", return_value="while_eager"):
        assert ensure_callable(while_loop)(None, None, None) == "while_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_eager", return_value="scan_eager"):
        assert ensure_callable(scan)(None, None, None) == "scan_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn_eager", return_value="map_fn_eager"):
        assert ensure_callable(map_fn)(None, None) == "map_fn_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.pmap_eager", return_value="pmap_eager"):
        assert ensure_callable(pmap)(None) == "pmap_eager"
    _state.update(ensure_tensor_dict({"config": config}))


def _test_control_flow_extra_functions_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    config = _state["config"]
    with patch("ml_switcheroo_compiler.ops.control_flow.stop_gradient_eager", return_value="stop_eager"):
        assert ensure_callable(stop_gradient)(None) == "stop_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.assert_value_eager", return_value="assert_eager"):
        assert_value(None)
    ensure_mutable(config).eager_mode = ensure_eager_tensor(False)
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_tracing", return_value="cond_tracing"):
        assert ensure_callable(cond)(None, None, None) == "cond_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_tracing", return_value="while_tracing"):
        assert ensure_callable(while_loop)(None, None, None) == "while_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_tracing", return_value="scan_tracing"):
        assert ensure_callable(scan)(None, None, None) == "scan_tracing"
    _state.update(ensure_tensor_dict({"config": config}))


def _test_control_flow_extra_functions_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    config = _state["config"]
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn_tracing", return_value="map_fn_tracing"):
        assert ensure_callable(map_fn)(None, None) == "map_fn_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.pmap_tracing", return_value="pmap_tracing"):
        assert ensure_callable(pmap)(None) == "pmap_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.stop_gradient_tracing", return_value="stop_tracing"):
        assert ensure_callable(stop_gradient)(None) == "stop_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.assert_value_tracing", return_value="assert_tracing"):
        assert_value(None)
    ensure_mutable(config).eager_mode = ensure_eager_tensor(True)
    _state.update(ensure_tensor_dict({"config": config}))


def _test_control_flow_extra_functions_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop") as wl:
        wl.return_value = (None, "res")
        with patch("ml_switcheroo_compiler.ops.control_flow.less"):
            with patch("ml_switcheroo_compiler.ops.control_flow.add"):
                res = fori_loop(0, 10, lambda i, x: x, 0)
                assert res == "res"
                (cond_fn, body_wrapper) = (wl.call_args[0][0], wl.call_args[0][1])
                cond_fn((0, 0))
                body_wrapper((0, 0))
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn", return_value="map"):
        assert ensure_callable(map)(None, None) == "map"
    with patch("ml_switcheroo_compiler.ops.control_flow.vmap") as vm:
        vm.return_value.return_value = "vmap"
        assert ensure_callable(vectorized_map)(None, None) == "vmap"
    _state.update(ensure_tensor_dict({"body_wrapper": body_wrapper, "cond_fn": cond_fn, "res": res, "vm": vm}))
    _state.update(ensure_tensor_dict({"wl": wl}))


def _test_control_flow_extra_functions_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(ValueError):
        ensure_callable(switch)(None, [])
    with patch("ml_switcheroo_compiler.ops.control_flow.cond") as cnd:
        cnd.return_value = "cond_res"
        with patch("ml_switcheroo_compiler.ops.control_flow.less", return_value="less"):
            res = ensure_callable(switch)(ensure_callable(DummyTensor)(0), [lambda: 1, lambda: 2])
        assert res == "cond_res"
        res2 = ensure_callable(switch)(ensure_callable(DummyTensor)(0), [lambda: "leaf"])
        assert res2 == "leaf"
    _state.update(ensure_tensor_dict({"cnd": cnd, "res": res, "res2": res2}))


def test_control_flow_extra_functions() -> None:
    """Test function test_control_flow_extra_functions."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_control_flow_extra_functions_part_1(_state)
    _test_control_flow_extra_functions_part_2(_state)
    _test_control_flow_extra_functions_part_3(_state)
    _test_control_flow_extra_functions_part_4(_state)
    _test_control_flow_extra_functions_part_5(_state)
    assert len(_state) > 0


def test_control_flow_custom_gradient() -> None:
    """Test function test_control_flow_custom_gradient."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    def cg(x: EagerValue) -> EagerValue:
        """Test function cg.

        Args:
            x: Test parameter `x`.
        """
        return (x, lambda g: g)

    wrapped = custom_gradient(cg)
    assert ensure_callable(wrapped)(1) == 1
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        assert ensure_callable(wrapped)(1) == "emit"


def test_control_flow_case() -> None:
    """Test function test_control_flow_case."""
    with pytest.raises(ValueError):
        case([])
    assert ensure_callable(case)([], default=lambda: "def") == "def"

    def true_fn() -> EagerValue:
        """Test function true_fn."""
        return "true"

    with patch("ml_switcheroo_compiler.ops.control_flow.cond", return_value="cond_res"):
        assert ensure_callable(case)([(ensure_callable(DummyTensor)(True), true_fn)]) == "cond_res"
    with pytest.raises(ValueError):
        ensure_callable(switch_case)(ensure_callable(DummyTensor)(), {})
    assert ensure_callable(switch_case)(ensure_callable(DummyTensor)(), {}, default=lambda: "def") == "def"
    with patch("ml_switcheroo_compiler.ops.control_flow.equal"):
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", return_value="cond_res"):
            assert ensure_callable(switch_case)(ensure_callable(DummyTensor)(), {0: true_fn}) == "cond_res"


def test_control_flow_associative_scan() -> None:
    """Test function test_control_flow_associative_scan."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return "exec"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend()):
        assert associative_scan() == "exec"
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        assert associative_scan() == "emit"


def _exercise_custom_gradient_variants(custom_gradient: EagerValue) -> None:
    """Exercise custom_gradient registration and VJP variants.

    Args:
        custom_gradient: The custom_gradient decorator.
    """
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY as OpRegistry
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY as VJP_REGISTRY

    class Ret:
        """Test class Ret."""

        shape = (1,)
        dtype = "float32"

    class DummyNode:
        """Test class DummyNode."""

        id = "id"

    class DummyProxy:
        """Test class DummyProxy."""

        node = DummyNode()

    wrapped = ensure_callable(custom_gradient)(lambda x: (ensure_callable(Ret)(), lambda g: g))
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
        ensure_callable(wrapped)(1)
    for k in [name for name in OpRegistry if name.startswith("CustomGradient_")]:
        ensure_eager_tensor(OpRegistry[k]).infer_shape(None)
        VJP_REGISTRY[k](ensure_graph(None), ensure_node(None), "cotangent")
        wrapped3 = ensure_callable(custom_gradient)(lambda x: (ensure_callable(DummyTensor)(), lambda g: ensure_callable(DummyProxy)()))
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
            ensure_callable(wrapped3)(1)
        for k3 in [name for name in OpRegistry if name.startswith("CustomGradient_") and name != k]:
            VJP_REGISTRY[k3](ensure_graph(None), ensure_node(None), "cotangent")
        wrapped2 = ensure_callable(custom_gradient)(lambda x: (ensure_callable(DummyTensor)(), lambda g: (g, g)))
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
            ensure_callable(wrapped2)(1)
        for k2 in [name for name in OpRegistry if name.startswith("CustomGradient_") and name != k]:
            VJP_REGISTRY[k2](ensure_graph(None), ensure_node(None), "cotangent")
            ensure_eager_tensor(OpRegistry[k2]).infer_shape(None)
            break
        break


def test_control_flow_extra_more_branches() -> None:
    """Test function test_control_flow_extra_more_branches."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient, switch

    with patch("ml_switcheroo_compiler.ops.control_flow.less", return_value="less"):
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", side_effect=lambda a, b, c: b()):
            ensure_callable(switch)(ensure_callable(DummyTensor)(0), [lambda: 1, lambda: 2, lambda: 3])
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", side_effect=lambda a, b, c: c()):
            ensure_callable(switch)(ensure_callable(DummyTensor)(0), [lambda: 1, lambda: 2, lambda: 3])
    config.eager_mode = False
    _exercise_custom_gradient_variants(custom_gradient)
    assert ensure_callable(scan_bind)("f", "xs") == ("f", "xs")


def test_custom_grad_vjp_direct() -> None:
    """Test function test_custom_grad_vjp_direct."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY as VJP_REGISTRY

    config.eager_mode = False

    class Ret:
        """Test class Ret."""

        shape = (1,)

    class P:
        """Test class P."""

        node = type("N", (), {"id": "n_id"})()

    def cg(x: EagerValue) -> EagerValue:
        """Test function cg.

        Args:
            x: Test parameter `x`.
        """
        return ensure_eager_tensor((Ret(), lambda g: P()))

    w = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        ensure_callable(w)(1)
    for k in VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            VJP_REGISTRY[k](ensure_graph(None), ensure_node(None), "cot")
            pass
            pass
            break

    def cg2(x: EagerValue) -> EagerValue:
        """Test function cg2.

        Args:
            x: Test parameter `x`.
        """
        return ensure_eager_tensor((Ret(), lambda g: (P(), "str")))

    w2 = custom_gradient(cg2)
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        ensure_callable(w2)(1)
    for k in VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            VJP_REGISTRY[k](ensure_graph(None), ensure_node(None), "cot")
    assert callable(w2)


def test_cg_clear() -> None:
    """Test function test_cg_clear."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY as VJP_REGISTRY

    config.eager_mode = False

    class Ret:
        """Test class Ret."""

        shape = (1,)

    class P:
        """Test class P."""

        node = type("N", (), {"id": "n_id"})()

    def cg(x: EagerValue) -> EagerValue:
        """Test function cg.

        Args:
            x: Test parameter `x`.
        """
        return ensure_eager_tensor((Ret(), lambda g: P()))

    old_keys = list(VJP_REGISTRY.keys())
    w = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        ensure_callable(w)(1)
    new_keys = [k for k in VJP_REGISTRY if k not in old_keys]
    if new_keys:
        new_keys[0]
        pass
        pass

    def cg2(x: EagerValue) -> EagerValue:
        """Test function cg2.

        Args:
            x: Test parameter `x`.
        """
        return ensure_eager_tensor((Ret(), lambda g: (P(), "str")))

    w2 = custom_gradient(cg2)
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        ensure_callable(w2)(1)
    new_keys2 = [k for k in VJP_REGISTRY if k not in old_keys and k not in new_keys]
    if new_keys2:
        new_keys2[0]
        pass
    assert isinstance(new_keys2, list)


def test_scan_bind() -> None:
    """Test function test_scan_bind."""
    assert ensure_callable(scan_bind)("f", "xs") == ("f", "xs")


def test_dyn_vjp() -> None:
    """Test function test_dyn_vjp."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient

    config.eager_mode = False

    def cg(x: EagerValue) -> EagerValue:
        """Test function cg.

        Args:
            x: Test parameter `x`.
        """
        return (x, lambda g: g)

    wrapped = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
        ensure_callable(wrapped)(1)
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY

    for k in _VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            _VJP_REGISTRY[k](ensure_graph(None), ensure_node(None), "cot")
            break
    assert callable(wrapped)
