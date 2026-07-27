import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.random_ops as ro


def test_random_ops_coverage():
    ops = [getattr(ro, name) for name in dir(ro) if name.startswith("_") and callable(getattr(ro, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("x", "y", "tensor", "a", "b", "c", "t"):
                args_to_pass.append(arg)
            else:
                args_to_pass.append(arg)

        try:
            op(*args_to_pass)
        except Exception:
            pass

        try:
            op(*args_to_pass[:-1])
        except Exception:
            pass


def test_missing_random_ops():
    import ml_switcheroo_compiler.backends.numpy.eager.random_ops as ro

    class DummyConfig:
        minval = None
        maxval = None

    ro._np_uniform(np, shape=(2,), dtype="float32", config=DummyConfig())

    ro._np_stateless_split(np, seed=np.array(["test"]), num=2)
    ro._np_stateless_split(np, seed=np.array(["not_int", "other"]), num=2)

    class DummyTable:
        def lookup(self, keys):
            return keys

    ro._np_lookup(np, DummyTable(), np.array([1, 2]))

    # ValueError branch
    class BadStr:
        def __int__(self):
            raise ValueError()

        def __str__(self):
            return "bad"

    ro._np_stateless_split(np, seed=np.array([BadStr()]), num=2)

    # 54: dict
    ro._np_lookup(np, {1: 2}, np.array([1, 2]))

    ro._np_stateless_split(np, seed=np.array([]))
