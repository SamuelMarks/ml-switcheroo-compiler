def test_frontend_utils_missing():
    from ml_switcheroo_compiler.ops.creation.frontend_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    # 1. _multinomial_shape where size is int
    class MockTensor:
        shape = (10,)

    assert _multinomial_shape(MockTensor(), size=5) == (5,)

    # 2. frompyfunc
    try:
        frompyfunc(lambda x: x, 1, 1)
    except Exception:
        pass
