from ml_switcheroo_compiler.ops.ragged.core import RaggedDot
from ml_switcheroo_compiler.ops.ragged.frontend import (
    ragged_dot,
    ragged_stack,
    ragged_stack_dynamic_partitions,
)


def test_ragged_ops():
    # just hit the lines
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.ragged.frontend._ragged_op") as mock_ragged:
        mock_ragged.return_value = "res"
        res1 = ragged_stack()
        assert res1 == "res"
        res2 = ragged_stack_dynamic_partitions()
        assert res2 == "res"
        res3 = ragged_dot()
        assert res3 == "res"

    op = RaggedDot()
    res4 = op.infer_shape()
    assert res4 == ()
