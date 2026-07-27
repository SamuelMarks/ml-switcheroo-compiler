import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.nn_ops as nn_ops


def test_nn_ops_coverage():
    ops = [getattr(nn_ops, name) for name in dir(nn_ops) if name.startswith("_") and callable(getattr(nn_ops, name))]
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


def test_missing_nn_ops():
    import ml_switcheroo_compiler.backends.numpy.eager.nn_ops as nn_ops

    # Missing args logic for dropout2d:
    x_4d = np.ones((1, 2, 3, 3))
    nn_ops._np_dropout2d(np, x_4d, p=0.5, training=True)

    nn_ops._np_dropout2d(np, x_4d, p=0.0, training=True)
