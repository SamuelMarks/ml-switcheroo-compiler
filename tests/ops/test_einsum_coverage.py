"""Module docstring."""

import pytest

from ml_switcheroo_compiler.ops.linalg.einsum import Einsum


def test_einsum_infer_shape_coverage() -> object:
    """Function docstring."""
    op = Einsum()

    # 1. Provide kwargs without equation
    with pytest.raises(ValueError, match="Einsum requires an 'equation' string attribute"):
        op.infer_shape((2, 3), (3, 4))

    # 2. Provide non-string equation
    with pytest.raises(ValueError, match="Einsum requires an 'equation' string attribute"):
        op.infer_shape(123, (2, 3), (3, 4))

    # 3. No shapes provided
    assert op.infer_shape("ij,jk->ik") == ()

    # 4. Implicit mode
    assert op.infer_shape("ij,jk", (2, 3), (3, 4)) == (2, 4)
    assert op.infer_shape("...ij,...jk", (5, 2, 3), (5, 3, 4)) == (5, 2, 4)

    # 5. Length mismatch of operands
    with pytest.raises(ValueError, match="Equation expected 2 inputs but got 1"):
        op.infer_shape("ij,jk->ik", (2, 3))

    # 6. Multiple ellipses in operand
    with pytest.raises(ValueError, match=r"Shape \(5, 2, 3\) cannot match subscript ...i...j"):
        op.infer_shape("...i...j,...jk->...ik", (5, 2, 3), (5, 3, 4))

    # 7. Shape \(5,\) cannot match subscript \.\.\.ijk
    with pytest.raises(ValueError, match=r"Shape \(5,\) cannot match subscript \.\.\.ijk"):
        op.infer_shape("...ijk,...jk->...ik", (5,), (5, 3, 4))

    # 8. Dimension mismatch for subscript (left of ellipsis)
    with pytest.raises(ValueError, match="Dimension mismatch for axis i"):
        op.infer_shape("i...,i...->...", (2,), (3,))

    # 9. Dimension mismatch for subscript (right of ellipsis)
    with pytest.raises(ValueError, match="Dimension mismatch for axis j"):
        op.infer_shape("...j,...j->...", (2,), (3,))

    # 10. Dimension mismatch (no ellipsis)
    with pytest.raises(ValueError, match="Dimension mismatch for axis j"):
        op.infer_shape("ij,jk->ik", (2, 3), (4, 4))

    # 11. Ellipsis broadcasting success
    assert op.infer_shape("...ij,...jk->...ik", (1, 2, 3), (5, 3, 4)) == (5, 2, 4)

    # 11.5 Ellipsis broadcasting success (d2 == 1)
    assert op.infer_shape("...ij,...jk->...ik", (5, 2, 3), (1, 3, 4)) == (5, 2, 4)

    # 12. Ellipsis broadcasting failure
    with pytest.raises(ValueError, match="Ellipsis shapes cannot be broadcast"):
        op.infer_shape("...ij,...jk->...ik", (2, 2, 3), (5, 3, 4))

    # 13. Shape \(2, 3, 1\) cannot match subscript ij (no ellipsis)
    with pytest.raises(ValueError, match=r"Shape \(2, 3, 1\) cannot match subscript ij"):
        op.infer_shape("ij,jk->ik", (2, 3, 1), (3, 4))

    # 14. Multiple ellipses in output
    with pytest.raises(ValueError, match="Multiple ellipses in output subscript"):
        op.infer_shape("...ij,...jk->...i...k", (5, 2, 3), (5, 3, 4))

    # 14.5 Left part in output ellipsis
    assert op.infer_shape("a...ij,a...jk->a...ik", (2, 5, 2, 3), (2, 5, 3, 4)) == (2, 5, 2, 4)

    # 15. Output subscript not in input
    with pytest.raises(ValueError, match="Output character l not found in inputs"):
        op.infer_shape("ij,jk->il", (2, 3), (3, 4))

    # 16. Skip None shapes (like *operands handling None)
    assert op.infer_shape("ij,jk->ik", (2, 3), None, (3, 4)) == (2, 4)

    # 17. Skip non-tuple shapes correctly and just construct map from the tuple ones
    assert op.infer_shape("i", "not a shape") == ()
