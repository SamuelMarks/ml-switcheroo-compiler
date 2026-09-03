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
