import pytest
from ml_switcheroo_compiler.ops.linalg.basic import EinsumEquationParser


def test_calculate_ellipsis_expansion():
    num, curr = EinsumEquationParser._calculate_ellipsis_expansion("i", "j", (2, 3, 4))
    assert num == 1
    assert curr == (3,)

    with pytest.raises(ValueError, match="Shape too small"):
        EinsumEquationParser._calculate_ellipsis_expansion("ij", "k", (2, 3))
