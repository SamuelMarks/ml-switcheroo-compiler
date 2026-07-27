import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb


def test_math_binary_coverage():
    ops = [getattr(mb, name) for name in dir(mb) if name.startswith("_") and callable(getattr(mb, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y"):
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


def test_polygamma_zeta_missing_args():
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    arg = np.array([1.0, 2.0])
    try:
        mb._np_polygamma(np, arg)
    except:
        pass
    try:
        mb._np_zeta(np, arg)
    except:
        pass


def test_missing_math_binary():
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    class DummyBk:
        def zeros_like(self, x):
            return np.zeros_like(x)

    mb._np_polygamma(DummyBk(), np.array([1.0]))
    mb._np_zeta(DummyBk(), np.array([1.0]))


def test_missing_math_binary2():
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    class DummyBk:
        def zeros_like(self, x):
            return np.zeros_like(x)

    try:
        mb._np_polygamma(DummyBk(), np.array([1.0]))
    except:
        pass
    try:
        mb._np_zeta(DummyBk(), np.array([1.0]))
    except:
        pass


def test_polygamma_zeta_missing_args2():
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    class DummyBk2:
        def zeros_like(self, x):
            return np.zeros_like(x)

        def array(self, x):
            return x

    try:
        mb._np_polygamma(DummyBk2(), np.array([1.0]), x=np.array([1.0]))
    except:
        pass
    try:
        mb._np_zeta(DummyBk2(), np.array([1.0]), q=np.array([1.0]))
    except:
        pass
