"""Tests for eager registry custom JVP/VJP derivatives and tuple dispatch."""

from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends import eager_registry as m


def test_tuple_get_item_coverage():
    backend = MagicMock()
    res = m._eager_tuple_get_item(backend, [10, 20], index=1)
    assert res == 20


def test_process_custom_vjp_call():
    backend = MagicMock()
    mock_bwd = MagicMock()
    mock_bwd.__call__ = MagicMock(return_value="res")
    res = m._eager_process_custom_vjp_call(backend, 5, bwd_fn=mock_bwd)
    assert res == "res"


def test_custom_jvp_and_process_custom_jvp_call():
    backend = MagicMock()

    # 1. _eager_custom_jvp with callable fun
    def my_fun(x, y):
        return x * y

    res_fun = m._eager_custom_jvp(backend, 3, 4, fun=my_fun)
    assert res_fun == 12

    # Test _eager_custom_jvp without callable fun
    assert m._eager_custom_jvp(backend, 42) == 42
    assert m._eager_custom_jvp(backend, 1, 2) == (1, 2)

    # 2. _eager_process_custom_jvp_call with 2-tuple return
    def rule_packed(primals, tangents):
        return (primals[0] * 2, tangents[0] * 2)

    res_rule1 = m._eager_process_custom_jvp_call(backend, 5, 1, num_primals=1, jvp_rule=rule_packed)
    assert res_rule1 == (10, 2)

    # 3. _eager_process_custom_jvp_call with unpacked fallback on TypeError
    def rule_unpacked(p, t):
        return (p + 1, t + 1)

    res_rule2 = m._eager_process_custom_jvp_call(backend, 10, 2, num_primals=1, jvp_rule=rule_unpacked)
    assert res_rule2 == (11, 3)

    # 4. _eager_process_custom_jvp_call returning non-tuple (falls back to (None, res))
    def rule_single(primals, tangents):
        return 999

    res_rule3 = m._eager_process_custom_jvp_call(backend, 1, 1, num_primals=1, jvp_rule=rule_single)
    assert res_rule3 == (None, 999)

    # 5. _eager_process_custom_jvp_call with non-callable jvp_rule
    res_none = m._eager_process_custom_jvp_call(backend, 1, 1, jvp_rule="not_callable")
    assert res_none == (None, None)
