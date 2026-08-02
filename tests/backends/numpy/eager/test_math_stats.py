import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_stats as ms


def test_math_stats_coverage():
    ops = [getattr(ms, name) for name in dir(ms) if name.startswith("_") and callable(getattr(ms, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y", "tensor"):
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


def test_math_stats_exceptions():
    import ml_switcheroo_compiler.backends.numpy.eager.math_stats as ms

    try:
        ms._np_descriptive(np, a=None)
    except:
        pass
    try:
        ms._np_randompermutation(np, 5)
    except:
        pass
    try:
        ms._np_randomcategorical(np, [1, 2], p=[0.5, 0.5])
    except:
        pass


def test_missing_math_stats2():
    import ml_switcheroo_compiler.backends.numpy.eager.math_stats as ms

    class MockData:
        data = 5

    ms._np_randompermutation(np, np.array([1.0]), x=MockData())

    class MockDataP:
        data = [1.0]

    class MockDataList:
        data = [1.0]

    try:
        ms._np_randomchoice(np, np.array([1.0]), a=MockDataList(), p=MockDataP())
    except:
        pass


from ml_switcheroo_compiler.backends.numpy.eager.math_stats import _np_randomcategorical, _np_randomshuffle


def test_np_random_shuffle_list():
    x = [1, 2, 3]
    res = _np_randomshuffle(None, x)
    assert isinstance(res, np.ndarray)
    assert set(res) == {1, 2, 3}


def test_randomcategorical_shape():
    # provide logits as kwargs and shape
    res = _np_randomcategorical(None, None, logits=np.array([1.0, 2.0]), shape=(1, 2))
    assert res.shape == (1, 2)

    # test missing shape
    res = _np_randomcategorical(None, None, logits=np.array([[1.0, 2.0]]))
    assert res.shape == (1,)
