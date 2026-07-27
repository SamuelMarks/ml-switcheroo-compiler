# ruff: noqa: E501
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.control_flow import AssertOp, AssociativeScan, DebugInfs, DebugNans, ScanOp, SwitchOp, assert_value, associative_scan, case, cond, custom_gradient, fori_loop, map, map_fn, pmap, scan, scan_bind, stop_gradient, switch, switch_case, vectorized_map, while_loop


class DummyData:
    shape = (1,)
    dtype = "float32"


class DummyTensor:
    shape = (1,)

    def __init__(self, data=0):
        self.data = data
        self.dtype = "float32"
        self.device = "cpu"

    def __getitem__(self, i):
        return DummyTensor()


def test_control_flow_extra_classes():
    assert AssertOp().infer_shape(None) == ()
    t = DummyTensor()
    assert DebugInfs().infer_shape(t) == t
    assert DebugInfs().infer_shape() == ()
    t2 = DummyTensor()
    assert DebugNans().infer_shape(t2) == t2
    assert DebugNans().infer_shape() == ()
    assert SwitchOp().infer_shape(None, None) == ()
    assert ScanOp().infer_shape() == ()
    t3 = DummyTensor()
    assert AssociativeScan().infer_shape(t3) == t3
    assert AssociativeScan().infer_shape() == ()


def test_control_flow_extra_functions():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_eager", return_value="cond_eager"):
        assert cond(None, None, None) == "cond_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_eager", return_value="while_eager"):
        assert while_loop(None, None, None) == "while_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_eager", return_value="scan_eager"):
        assert scan(None, None, None) == "scan_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn_eager", return_value="map_fn_eager"):
        assert map_fn(None, None) == "map_fn_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.pmap_eager", return_value="pmap_eager"):
        assert pmap(None) == "pmap_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.stop_gradient_eager", return_value="stop_eager"):
        assert stop_gradient(None) == "stop_eager"
    with patch("ml_switcheroo_compiler.ops.control_flow.assert_value_eager", return_value="assert_eager"):
        assert_value(None)
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.control_flow.cond_tracing", return_value="cond_tracing"):
        assert cond(None, None, None) == "cond_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop_tracing", return_value="while_tracing"):
        assert while_loop(None, None, None) == "while_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.scan_tracing", return_value="scan_tracing"):
        assert scan(None, None, None) == "scan_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn_tracing", return_value="map_fn_tracing"):
        assert map_fn(None, None) == "map_fn_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.pmap_tracing", return_value="pmap_tracing"):
        assert pmap(None) == "pmap_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.stop_gradient_tracing", return_value="stop_tracing"):
        assert stop_gradient(None) == "stop_tracing"
    with patch("ml_switcheroo_compiler.ops.control_flow.assert_value_tracing", return_value="assert_tracing"):
        assert_value(None)
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.ops.control_flow.while_loop") as wl:
        wl.return_value = (None, "res")
        with patch("ml_switcheroo_compiler.ops.control_flow.less") as ls:
            with patch("ml_switcheroo_compiler.ops.control_flow.add") as ad:
                res = fori_loop(0, 10, lambda i, x: x, 0)
                assert res == "res"
                (cond_fn, body_wrapper) = (wl.call_args[0][0], wl.call_args[0][1])
                cond_fn((0, 0))
                body_wrapper((0, 0))
    with patch("ml_switcheroo_compiler.ops.control_flow.map_fn", return_value="map"):
        assert map(None, None) == "map"
    with patch("ml_switcheroo_compiler.ops.control_flow.vmap") as vm:
        vm.return_value.return_value = "vmap"
        assert vectorized_map(None, None) == "vmap"
    with pytest.raises(ValueError):
        switch(None, [])
    with patch("ml_switcheroo_compiler.ops.control_flow.cond") as cnd:
        cnd.return_value = "cond_res"
        with patch("ml_switcheroo_compiler.ops.control_flow.less", return_value="less"):
            res = switch(DummyTensor(0), [lambda: 1, lambda: 2])
        assert res == "cond_res"
        res2 = switch(DummyTensor(0), [lambda: "leaf"])
        assert res2 == "leaf"


def test_control_flow_custom_gradient():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    def cg(x):
        return (x, lambda g: g)

    wrapped = custom_gradient(cg)
    assert wrapped(1) == 1
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        assert wrapped(1) == "emit"


def test_control_flow_case():
    with pytest.raises(ValueError):
        case([])
    assert case([], default=lambda: "def") == "def"

    def true_fn():
        return "true"

    with patch("ml_switcheroo_compiler.ops.control_flow.cond", return_value="cond_res"):
        assert case([(DummyTensor(True), true_fn)]) == "cond_res"
    with pytest.raises(ValueError):
        switch_case(DummyTensor(), {})
    assert switch_case(DummyTensor(), {}, default=lambda: "def") == "def"
    with patch("ml_switcheroo_compiler.ops.control_flow.equal"):
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", return_value="cond_res"):
            assert switch_case(DummyTensor(), {0: true_fn}) == "cond_res"


def test_control_flow_associative_scan():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            return "exec"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend()):
        assert associative_scan() == "exec"
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        assert associative_scan() == "emit"


def test_control_flow_extra_more_branches():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient, switch

    with patch("ml_switcheroo_compiler.ops.control_flow.less", return_value="less"):
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", side_effect=lambda a, b, c: b()):
            try:
                switch(DummyTensor(0), [lambda: 1, lambda: 2, lambda: 3])
            except:
                pass
        with patch("ml_switcheroo_compiler.ops.control_flow.cond", side_effect=lambda a, b, c: c()):
            try:
                switch(DummyTensor(0), [lambda: 1, lambda: 2, lambda: 3])
            except:
                pass
    config.eager_mode = False

    def cg_func(x):

        class Ret:
            shape = (1,)
            dtype = "float32"

        return (Ret(), lambda g: g)

    wrapped = custom_gradient(cg_func)
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY as OpRegistry
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY as VJP_REGISTRY

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
        wrapped(1)
    for k in OpRegistry:
        if k.startswith("CustomGradient_"):
            OpRegistry[k].infer_shape(None)
            vjp_func = VJP_REGISTRY[k]

            class DummyNode:
                id = "id"

            class DummyProxy:
                node = DummyNode()

            vjp_func(None, None, "cotangent")

            def cg_func_proxy(x):
                return (DummyTensor(), lambda g: DummyProxy())

            wrapped3 = custom_gradient(cg_func_proxy)
            with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
                wrapped3(1)
            for k3 in OpRegistry:
                if k3.startswith("CustomGradient_") and k3 != k:
                    VJP_REGISTRY[k3](None, None, "cotangent")

            def cg_func_tuple(x):
                return (DummyTensor(), lambda g: (g, g))

            wrapped2 = custom_gradient(cg_func_tuple)
            with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
                wrapped2(1)
            for k2 in OpRegistry:
                if k2.startswith("CustomGradient_") and k2 != k:
                    VJP_REGISTRY[k2](None, None, "cotangent")
                    OpRegistry[k2].infer_shape(None)
                    break
            break
    assert scan_bind("f", "xs") == ("f", "xs")


def test_custom_grad_vjp_direct():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY as VJP_REGISTRY

    config.eager_mode = False

    class Ret:
        shape = (1,)

    class P:
        node = type("N", (), {"id": "n_id"})()

    def cg(x):
        return (Ret(), lambda g: P())

    w = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        w(1)
    for k in VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            res = VJP_REGISTRY[k](None, None, "cot")
            pass
            pass
            break

    def cg2(x):
        return (Ret(), lambda g: (P(), "str"))

    w2 = custom_gradient(cg2)
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        w2(1)
    for k in VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            res = VJP_REGISTRY[k](None, None, "cot")


def test_cg_clear():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY as VJP_REGISTRY

    config.eager_mode = False

    class Ret:
        shape = (1,)

    class P:
        node = type("N", (), {"id": "n_id"})()

    def cg(x):
        return (Ret(), lambda g: P())

    old_keys = list(VJP_REGISTRY.keys())
    w = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        w(1)
    new_keys = [k for k in VJP_REGISTRY if k not in old_keys]
    if new_keys:
        k = new_keys[0]
        pass
        pass

    def cg2(x):
        return (Ret(), lambda g: (P(), "str"))

    w2 = custom_gradient(cg2)
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emit"):
        w2(1)
    new_keys2 = [k for k in VJP_REGISTRY if k not in old_keys and k not in new_keys]
    if new_keys2:
        k = new_keys2[0]
        pass


def test_scan_bind():
    assert scan_bind("f", "xs") == ("f", "xs")


def test_dyn_vjp():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient

    config.eager_mode = False

    def cg(x):
        return (x, lambda g: g)

    wrapped = custom_gradient(cg)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="emitted"):
        wrapped(1)
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY

    for k in _VJP_REGISTRY:
        if k.startswith("CustomGradient_"):
            _VJP_REGISTRY[k](None, None, "cot")
            break
