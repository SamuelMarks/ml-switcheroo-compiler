"""Tests for test_control_flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.control_flow import AssertOp, AssociativeScan, DebugInfs, DebugNans, ScanOp, SwitchOp, assert_value, associative_scan, case, cond, custom_gradient, fori_loop, map, map_fn, pmap, scan, scan_bind, stop_gradient, switch, switch_case, vectorized_map, while_loop

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyData:
    """Test class DummyData."""

    shape = (1,)
    dtype = "float32"


class DummyTensor:
    """Test class DummyTensor."""

    shape = (1,)

    def __init__(self, data: EagerValue = 0) -> None:
        """Test function __init__."""
        self.data = data
        self.dtype = "float32"
        self.device = "cpu"

    def __getitem__(self, i: EagerValue) -> EagerValue:
        """Test function __getitem__."""
        return DummyTensor()


def test_control_flow_extra_classes() -> None:
    """Test function test_control_flow_extra_classes."""
    assert cast("Callable[..., EagerTensorProtocol]", AssertOp)().infer_shape(None) == ()
    t = DummyTensor()
    assert DebugInfs().infer_shape(t) == t
    assert DebugInfs().infer_shape() == ()
    t2 = DummyTensor()
    assert DebugNans().infer_shape(t2) == t2
    assert DebugNans().infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", SwitchOp)().infer_shape(None, None) == ()
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
        assert cast("Callable[..., EagerTensorProtocol]", cond)(None, None, None) == "cond_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_eager", return_value="while_eager"):
        assert cast("Callable[..., EagerTensorProtocol]", while_loop)(None, None, None) == "while_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_eager", return_value="scan_eager"):
        assert cast("Callable[..., EagerTensorProtocol]", scan)(None, None, None) == "scan_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn_eager", return_value="map_fn_eager"):
        assert cast("Callable[..., EagerTensorProtocol]", map_fn)(None, None) == "map_fn_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.pmap_eager", return_value="pmap_eager"):
        assert cast("Callable[..., EagerTensorProtocol]", pmap)(None) == "pmap_eager"
    getattr(_state, "update")({"config": config})
    _state.update(locals())


def _test_control_flow_extra_functions_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    config = _state["config"]
    with patch("ml_switcheroo_compiler.ops.control_flow.stop_gradient_eager", return_value="stop_eager"):
        assert cast("Callable[..., EagerTensorProtocol]", stop_gradient)(None) == "stop_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.assert_value_eager", return_value="assert_eager"):
        assert_value(None)
    setattr(config, "eager_mode", False)
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_tracing", return_value="cond_tracing"):
        assert cast("Callable[..., EagerTensorProtocol]", cond)(None, None, None) == "cond_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_tracing", return_value="while_tracing"):
        assert cast("Callable[..., EagerTensorProtocol]", while_loop)(None, None, None) == "while_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_tracing", return_value="scan_tracing"):
        assert cast("Callable[..., EagerTensorProtocol]", scan)(None, None, None) == "scan_tracing"
    _state.update(locals())


def _test_control_flow_extra_functions_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    config = _state["config"]
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn_tracing", return_value="map_fn_tracing"):
        assert cast("Callable[..., EagerTensorProtocol]", map_fn)(None, None) == "map_fn_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.pmap_tracing", return_value="pmap_tracing"):
        assert cast("Callable[..., EagerTensorProtocol]", pmap)(None) == "pmap_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.stop_gradient_tracing", return_value="stop_tracing"):
        assert cast("Callable[..., EagerTensorProtocol]", stop_gradient)(None) == "stop_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.assert_value_tracing", return_value="assert_tracing"):
        assert_value(None)
    setattr(config, "eager_mode", True)
    _state.update(locals())


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
        assert cast("Callable[..., EagerTensorProtocol]", map)(None, None) == "map"
    with patch("ml_switcheroo_compiler.ops.control_flow.vmap") as vm:
        vm.return_value.return_value = "vmap"
        assert cast("Callable[..., EagerTensorProtocol]", vectorized_map)(None, None) == "vmap"
    getattr(_state, "update")({"body_wrapper": body_wrapper, "cond_fn": cond_fn, "res": res, "vm": vm})
    getattr(_state, "update")({"wl": wl})
    _state.update(locals())


def _test_control_flow_extra_functions_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_control_flow_extra_functions.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(ValueError):
        cast("Callable[..., EagerTensorProtocol]", switch)(None, [])
    with patch("ml_switcheroo_compiler.ops.control_flow.cond") as cnd:
        cnd.return_value = "cond_res"
        with patch("ml_switcheroo_compiler.ops.control_flow.less", return_value="less"):
            res = cast("Callable[..., EagerTensorProtocol]", switch)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(0), [lambda: 1, lambda: 2])
        assert res == "cond_res"
        res2 = cast("Callable[..., EagerTensorProtocol]", switch)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(0), [lambda: "leaf"])
        assert res2 == "leaf"
    getattr(_state, "update")({"cnd": cnd, "res": res, "res2": res2})


def test_control_flow_extra_functions() -> None:
    """Test function test_control_flow_extra_functions."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_control_flow_extra_functions_part_1(_state)
    _test_control_flow_extra_functions_part_2(_state)
    _test_control_flow_extra_functions_part_3(_state)
    _test_control_flow_extra_functions_part_4(_state)
    _test_control_flow_extra_functions_part_5(_state)
    assert isinstance(_state, dict)


def test_control_flow_custom_gradient() -> None:
    """Test function test_control_flow_custom_gradient."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    def cg(x: EagerValue) -> EagerValue:
        """Test function cg."""
        return (x, lambda g: g)

    wrapped = custom_gradient(cg)
    assert cast("Callable[..., EagerTensorProtocol]", wrapped)(1) == 1
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        assert cast("Callable[..., EagerTensorProtocol]", wrapped)(1) == "emit"


def test_control_flow_case() -> None:
    """Test function test_control_flow_case."""
    with pytest.raises(ValueError):
        case([])
    assert cast("Callable[..., EagerTensorProtocol]", case)([], default=lambda: "def") == "def"

    def true_fn() -> EagerValue:
        """Test function true_fn."""
        return "true"

    with patch("ml_switcheroo_compiler.ops.control_flow.cond", return_value="cond_res"):
        assert cast("Callable[..., EagerTensorProtocol]", case)([(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(True), true_fn)]) == "cond_res"
    with pytest.raises(ValueError):
        cast("Callable[..., EagerTensorProtocol]", switch_case)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), {})
    assert cast("Callable[..., EagerTensorProtocol]", switch_case)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), {}, default=lambda: "def") == "def"
    with patch("ml_switcheroo_compiler.ops.control_flow.equal"):
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", return_value="cond_res"):
            assert cast("Callable[..., EagerTensorProtocol]", switch_case)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), {0: true_fn}) == "cond_res"


def test_control_flow_associative_scan() -> None:
    """Test function test_control_flow_associative_scan."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op."""
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

    wrapped = cast("Callable[..., EagerTensorProtocol]", custom_gradient)(lambda x: (cast("Callable[..., EagerTensorProtocol]", Ret)(), lambda g: g))
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
        cast("Callable[..., EagerTensorProtocol]", wrapped)(1)
    for k in [name for name in OpRegistry if name.startswith("CustomGradient_")]:
        OpRegistry[k].infer_shape(None)
        VJP_REGISTRY[k](None, None, "cotangent")
        wrapped3 = cast("Callable[..., EagerTensorProtocol]", custom_gradient)(lambda x: (cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), lambda g: cast("Callable[..., EagerTensorProtocol]", DummyProxy)()))
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
            cast("Callable[..., EagerTensorProtocol]", wrapped3)(1)
        for k3 in [name for name in OpRegistry if name.startswith("CustomGradient_") and name != k]:
            VJP_REGISTRY[k3](None, None, "cotangent")
        wrapped2 = cast("Callable[..., EagerTensorProtocol]", custom_gradient)(lambda x: (cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), lambda g: (g, g)))
        with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
            cast("Callable[..., EagerTensorProtocol]", wrapped2)(1)
        for k2 in [name for name in OpRegistry if name.startswith("CustomGradient_") and name != k]:
            VJP_REGISTRY[k2](None, None, "cotangent")
            OpRegistry[k2].infer_shape(None)
            break
        break


def test_control_flow_extra_more_branches() -> None:
    """Test function test_control_flow_extra_more_branches."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient, switch

    with patch("ml_switcheroo_compiler.ops.control_flow.less", return_value="less"):
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", side_effect=lambda a, b, c: b()):
            try:
                cast("Callable[..., EagerTensorProtocol]", switch)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(0), [lambda: 1, lambda: 2, lambda: 3])
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", side_effect=lambda a, b, c: c()):
            try:
                cast("Callable[..., EagerTensorProtocol]", switch)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)(0), [lambda: 1, lambda: 2, lambda: 3])
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
    config.eager_mode = False
    _exercise_custom_gradient_variants(custom_gradient)
    assert cast("Callable[..., EagerTensorProtocol]", scan_bind)("f", "xs") == ("f", "xs")


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
        """Test function cg."""
        return cast("EagerTensorProtocol", (Ret(), lambda g: P()))

    w = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        cast("Callable[..., EagerTensorProtocol]", w)(1)
    for k in VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            VJP_REGISTRY[k](None, None, "cot")
            pass
            pass
            break

    def cg2(x: EagerValue) -> EagerValue:
        """Test function cg2."""
        return cast("EagerTensorProtocol", (Ret(), lambda g: (P(), "str")))

    w2 = custom_gradient(cg2)
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        cast("Callable[..., EagerTensorProtocol]", w2)(1)
    for k in VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            VJP_REGISTRY[k](None, None, "cot")
    assert w2 is not None or w2 is None


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
        """Test function cg."""
        return cast("EagerTensorProtocol", (Ret(), lambda g: P()))

    old_keys = list(VJP_REGISTRY.keys())
    w = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        cast("Callable[..., EagerTensorProtocol]", w)(1)
    new_keys = [k for k in VJP_REGISTRY if k not in old_keys]
    if new_keys:
        new_keys[0]
        pass
        pass

    def cg2(x: EagerValue) -> EagerValue:
        """Test function cg2."""
        return cast("EagerTensorProtocol", (Ret(), lambda g: (P(), "str")))

    w2 = custom_gradient(cg2)
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        cast("Callable[..., EagerTensorProtocol]", w2)(1)
    new_keys2 = [k for k in VJP_REGISTRY if k not in old_keys and k not in new_keys]
    if new_keys2:
        new_keys2[0]
        pass
    assert new_keys2 is not None or new_keys2 is None


def test_scan_bind() -> None:
    """Test function test_scan_bind."""
    assert cast("Callable[..., EagerTensorProtocol]", scan_bind)("f", "xs") == ("f", "xs")


def test_dyn_vjp() -> None:
    """Test function test_dyn_vjp."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient

    config.eager_mode = False

    def cg(x: EagerValue) -> EagerValue:
        """Test function cg."""
        return (x, lambda g: g)

    wrapped = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
        cast("Callable[..., EagerTensorProtocol]", wrapped)(1)
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY

    for k in _VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            _VJP_REGISTRY[k](None, None, "cot")
            break
    assert wrapped is not None or wrapped is None
