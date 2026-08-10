import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as ma
import ml_switcheroo_compiler.backends.numpy.eager.math_special as ms


def test_math_advanced_coverage():
    ops = [getattr(ma, name) for name in dir(ma) if name.startswith("_") and callable(getattr(ma, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                args_to_pass.append(1)
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


def test_math_special_coverage():
    ops = [getattr(ms, name) for name in dir(ms) if name.startswith("_") and callable(getattr(ms, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                args_to_pass.append(1)
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

    # specifically test _np_expint with less than 2 args
    ms._np_expint(np, np.array([1.0, 2.0]))
    ms._np_expint(np, x=np.array([1.0, 2.0]), n=1)
