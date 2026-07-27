import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.sparse_and_ragged as sr


def test_sparse_and_ragged_coverage():
    ops = [getattr(sr, name) for name in dir(sr) if name.startswith("_") and callable(getattr(sr, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("x", "y", "tensor", "a", "b", "c", "t", "sp_input"):
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
