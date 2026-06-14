"""Module docstring."""

from ml_switcheroo_compiler.ops.shape.basic import TopK


def test_topk_infer_shape_k_forms() -> None:
    """Docstring."""
    op = TopK()

    # k with __array__
    class KArray:
        """Docstring."""

        def __array__(self) -> object:
            """Docstring."""
            return KItem()

    class KItem:
        """Docstring."""

        def item(self) -> int:
            """Docstring."""
            return 2

    class MockTensor:
        """Docstring."""

        shape = (10,)

    # test __array__ and item
    assert op.infer_shape(MockTensor(), k=KArray()) == (2,)

    # test int parsing fallback
    assert op.infer_shape(MockTensor(), k="3") == (3,)

    # test exception parsing
    res = op.infer_shape(MockTensor(), k=object())
    assert type(res[0]) is object


def test_topk_infer_shape_tuple_array() -> None:
    """Docstring."""
    op = TopK()

    class KTuple(tuple):
        """Docstring."""

        def __array__(self) -> object:
            """Docstring."""
            return 1

    class MockTensor:
        """Docstring."""

        shape = (10,)

    res = op.infer_shape(MockTensor(), k=KTuple())
    # Should fall back to try int(k) which fails, then Exception pass
    assert type(res[0]) is KTuple


def test_topk_infer_shape_k_none() -> None:
    """Docstring."""
    op = TopK()

    class MockTensor:
        """Docstring."""

        shape = (10,)

    assert op.infer_shape(MockTensor()) == (1,)
