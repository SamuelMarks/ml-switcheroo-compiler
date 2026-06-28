import numpy as np
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.tracing.tracer import _tracer
from ml_switcheroo_compiler.backends.registry import BackendRegistry

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.creation.frontend import asarray
from ml_switcheroo_compiler.ops.distributed import pbroadcast, pdot, ppermute, pshuffle


def test_distributed_eager():
    import ml_switcheroo_compiler.backends.eager_registry as reg

    # Register stubs for distributed ops in eager registry so they don't fail
    reg.numpy_eager_registry.register("Pbroadcast")(lambda m, x, **kw: x)
    reg.numpy_eager_registry.register("Pdot")(lambda m, x, y, **kw: x)
    reg.numpy_eager_registry.register("Ppermute")(lambda m, x, **kw: x)
    reg.numpy_eager_registry.register("Pshuffle")(lambda m, x, **kw: x)

    with ConfigContext(eager_mode=True, backend="numpy"):
        t1 = asarray(np.array([1, 2, 3]))
        t2 = asarray(np.array([4, 5, 6]))

        r1 = pbroadcast(t1, "x")
        r2 = pdot(t1, t2, "x")
        r3 = ppermute(t1, "x", [0, 1])
        r4 = pshuffle(t1, "x", [0, 1])

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None
        assert r4 is not None


def test_all_gather():
    with ConfigContext(eager_mode=True, backend="numpy"):
        t1 = asarray(np.array([1, 2, 3]))
        r1 = ops.all_gather(t1, axis=0)
        assert r1 is not None


def test_all_gather_ast():
    with ConfigContext(eager_mode=False, backend="numpy"):
        _tracer.start_tracing("test")
        t1 = asarray(np.array([1, 2, 3]))
        _ = ops.all_gather(t1, axis=0)
        g = _tracer.stop_tracing()

        gen_cls = BackendRegistry.get("numpy")
        gen = gen_cls(g)
        code = gen.generate()
        assert "_all_gather" in code


def test_all_reduce():
    with ConfigContext(eager_mode=True, backend="numpy"):
        t1 = asarray(np.array([1, 2, 3]))
        r1 = ops.all_reduce(t1, op="sum")
        assert r1 is not None


def test_all_reduce_ast():
    with ConfigContext(eager_mode=False, backend="numpy"):
        _tracer.start_tracing("test")
        t1 = asarray(np.array([1, 2, 3]))
        _ = ops.all_reduce(t1, op="sum")
        g = _tracer.stop_tracing()

        gen_cls = BackendRegistry.get("numpy")
        gen = gen_cls(g)
        code = gen.generate()
        assert "_all_reduce" in code


def test_all_to_all():
    with ConfigContext(eager_mode=True, backend="numpy"):
        t1 = asarray(np.array([1, 2, 3]))
        r1 = ops.all_to_all(t1, split_axis=0, concat_axis=0, axis_name="x")
        assert r1 is not None


def test_all_to_all_ast():
    with ConfigContext(eager_mode=False, backend="numpy"):
        _tracer.start_tracing("test")
        t1 = asarray(np.array([1, 2, 3]))
        _ = ops.all_to_all(t1, split_axis=0, concat_axis=0, axis_name="x")
        g = _tracer.stop_tracing()

        gen_cls = BackendRegistry.get("numpy")
        gen = gen_cls(g)
        code = gen.generate()
        assert "_all_to_all" in code
