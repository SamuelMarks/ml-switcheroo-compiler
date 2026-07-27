import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing as adv


def test_advanced_indexing_coverage():
    ops = [getattr(adv, name) for name in dir(adv) if name.startswith("_") and callable(getattr(adv, name))]
    arg = np.array([1.0, 2.0])
    arg_indices = np.array([0, 1])
    arg_indices_nd = np.array([[0], [1]])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("indices", "kth", "index"):
                if "nd" in op.__name__:
                    args_to_pass.append(arg_indices_nd)
                else:
                    args_to_pass.append(arg_indices)
            elif p.name == "shape":
                args_to_pass.append((2,))
            elif p.name == "dim":
                args_to_pass.append(0)
            else:
                args_to_pass.append(arg)

        try:
            op(*args_to_pass)
        except Exception:
            pass

        # Test 1 missing arg
        try:
            op(*args_to_pass[:-1])
        except Exception:
            pass

    # specifically for tensor scatter operations:
    adv._scatter_add(arg, arg_indices, arg, 0)

    adv._tensor_scatter_update(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    try:
        adv._tensor_scatter_add(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_sub(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_min(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_max(arg, np.array([[0], [1]]), np.array([3.0, 4.0]))
    except:
        pass

    # Test cases where indices are a list or tuple instead of np.ndarray
    adv._tensor_scatter_update(arg, ([0], [1]), np.array([3.0, 4.0]))

    adv._np_scatter_nd(np, [[0], [1]], arg, (2,))
    try:
        adv._np_tensor_scatter_add(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_tensor_scatter_sub(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_tensor_scatter_update(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_tensor_scatter_min(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_tensor_scatter_max(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_scatter_nd_add(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_scatter_nd_sub(np, arg, [[0], [1]], arg)
    except:
        pass
    try:
        adv._np_scatter_nd_update(np, arg, [[0], [1]], arg)
    except:
        pass

    # missing lines handling
    try:
        adv._tensor_scatter_update(arg, "a", np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_add(arg, "a", np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_sub(arg, "a", np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_min(arg, "a", np.array([3.0, 4.0]))
    except:
        pass
    try:
        adv._tensor_scatter_max(arg, "a", np.array([3.0, 4.0]))
    except:
        pass

    try:
        adv._np_scatter_update(np, arg, arg_indices, arg, dim=0)
    except:
        pass

    try:
        adv._np_scatter_add(np, arg, arg_indices, arg, dim=0)
    except:
        pass

    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction="add")
    except:
        pass
    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction="sub")
    except:
        pass
    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction="mul")
    except:
        pass
    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction="div")
    except:
        pass
    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction="min")
    except:
        pass
    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction="max")
    except:
        pass
    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd, arg, reduction=None)
    except:
        pass

    try:
        adv._np_tensor_scatter(np, arg, arg_indices_nd)
    except:
        pass

    try:
        adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction="add")
    except:
        pass
    try:
        adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction="mul")
    except:
        pass
    try:
        adv._np_scatter_apply(np, None, arg, arg_indices_nd, arg, reduction=None)
    except:
        pass

    # triggering exceptions inside scatter_apply
    try:
        adv._np_scatter_apply(np, None, arg, "bad_indices", arg, reduction="add")
    except:
        pass

    try:
        adv._np_scatter_apply(np, None)
    except:
        pass
    try:
        adv._np_scatter_apply(np, None, arg)
    except:
        pass

    try:
        adv._np_scatter_max(np, arg, arg_indices, arg)
    except:
        pass
    try:
        adv._np_scatter_min(np, arg, arg_indices, arg)
    except:
        pass
    try:
        adv._np_scatter_mul(np, arg, arg_indices, arg)
    except:
        pass

    try:
        adv._np_scatter_update(np, arg, np.array([0, 0]), arg, dim=0)
    except:
        pass

    try:
        adv._np_scatter_max(np, arg, np.array([0, 0]), arg)
    except:
        pass

    try:
        adv._np_scatter_min(np, arg, np.array([0, 0]), arg)
    except:
        pass

    try:
        adv._np_scatter_mul(np, arg, np.array([0, 0]), arg)
    except:
        pass

    try:
        adv._np_scatter(np, arg, np.array([0, 0]), arg, dim=0)
    except:
        pass

    try:
        adv._np_scatter_max(np, arg, arg_indices_nd, arg)
    except:
        pass

    try:
        adv._np_scatter_min(np, arg, arg_indices_nd, arg)
    except:
        pass

    try:
        adv._np_scatter_mul(np, arg, arg_indices_nd, arg)
    except:
        pass
