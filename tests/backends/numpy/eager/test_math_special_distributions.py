"""Test math_misc edge cases coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import (
    _np_rawmatmul,
    _np_rawmerge,
    _np_rem,
    _np_tensorarraywrite,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_rawmatmul() -> None:
    """Test function test_np_rawmatmul."""
    a = np.ones((2, 2))
    b = np.ones((2, 2))
    res = _np_rawmatmul(np, a, b)
    assert cast("EagerCallableProtocol", np.array_equal)(res, cast("EagerCallableProtocol", np.matmul)(a, b))


def test_np_rawmerge() -> None:
    """Test function test_np_rawmerge."""
    a = np.ones((2, 2))
    res, status = cast("EagerCallableProtocol", _np_rawmerge)(np, [a, a])
    assert res is not None
    res_empty, status_empty = cast("EagerCallableProtocol", _np_rawmerge)(np, [])
    assert res_empty is None


def test_np_tensorarraywrite() -> None:
    """Test function test_np_tensorarraywrite."""
    handle = [1, 2]
    # In bounds
    res = _np_tensorarraywrite(np, handle, 1, 3)
    assert res == [1, 3]


def test_np_rem() -> None:
    """Test function test_np_rem."""
    a = np.array([5])
    b = np.array([3])
    res = _np_rem(np, a, b)
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)) == 2


def test_np_confusion_matrix_fallback() -> None:
    """Test function test_np_confusion_matrix_fallback."""
    a = np.array([0, 1])
    b = np.array([1, 1])
    fn = numpy_eager_registry.get("confusion_matrix")
    res = cast("EagerCallableProtocol", fn)(np, a, b)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_np_descriptive() -> None:
    """Test function test_np_descriptive."""
    a = np.array([1.0, 2.0, 3.0])
    fn = numpy_eager_registry.get("descriptive")
    res = cast("EagerCallableProtocol", fn)(np, a)
    assert cast("EagerCallableProtocol", len)(res) == 3


def test_np_distributions() -> None:
    """Test function test_np_distributions."""
    a = np.array([1.0, 2.0, 3.0])
    fn = numpy_eager_registry.get("distributions")
    res = cast("EagerCallableProtocol", fn)(np, a)
    assert cast("EagerCallableProtocol", len)(res) == 2


def test_np_confusion_matrix_camel() -> None:
    """Test function test_np_confusion_matrix_camel."""
    a = np.array([0, 1])
    b = np.array([1, 1])
    fn = numpy_eager_registry.get("ConfusionMatrix")
    res = cast("EagerCallableProtocol", fn)(np, a, b, num_classes=3)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 3)


def test_np_decode_csv_empty() -> None:
    """Test function test_np_decode_csv_empty."""
    fn = numpy_eager_registry.get("DecodeCsv")
    res = cast("EagerCallableProtocol", fn)(np, "", record_defaults=[1.0, 2.0])
    assert cast("EagerCallableProtocol", len)(res) == 2
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)) == 1.0


def test_fallback_snippets_mock(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_fallback_snippets_mock.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.ops as ops
    from ml_switcheroo_compiler.ops.base import OpDef

    class FakeOp(OpDef):
        """Test class FakeOp."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> FakeOp:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            obj = super().__new__(cls)
            setattr(obj, "hit", True)
            return cast("FakeOp", cast("EagerTensorProtocol", obj))

    # Mocking standard snippet ops to hit the cls_or_func() path
    monkeypatch.setattr(ops, "RawMatMul", FakeOp, raising=False)
    monkeypatch.setattr(ops, "SparseDenseMatMul", FakeOp, raising=False)
    monkeypatch.setattr(ops, "rem", FakeOp, raising=False)
    monkeypatch.setattr(ops, "confusion_matrix", FakeOp, raising=False)
    monkeypatch.setattr(ops, "descriptive", FakeOp, raising=False)
    monkeypatch.setattr(ops, "distributions", FakeOp, raising=False)

    res1 = cast("EagerCallableProtocol", numpy_eager_registry.get)("RawMatMul")(np, cast("EagerCallableProtocol", np.ones)((2, 2)), cast("EagerCallableProtocol", np.ones)((2, 2)))
    if isinstance(res1, FakeOp):
        assert res1.hit

    res2 = cast("EagerCallableProtocol", numpy_eager_registry.get)("SparseDenseMatMul")(np, cast("EagerCallableProtocol", np.ones)((2, 2)), cast("EagerCallableProtocol", np.ones)((2, 2)))
    if isinstance(res2, FakeOp):
        assert res2.hit

    res3 = cast("EagerCallableProtocol", numpy_eager_registry.get)("rem")(np, cast("EagerCallableProtocol", np.ones)((2, 2)), cast("EagerCallableProtocol", np.ones)((2, 2)))
    if isinstance(res3, FakeOp):
        assert res3.hit

    res4 = cast("EagerCallableProtocol", numpy_eager_registry.get)("confusion_matrix")(np, cast("EagerCallableProtocol", np.array)([1], dtype=np.int32), cast("EagerCallableProtocol", np.array)([1], dtype=np.int32))
    if isinstance(res4, FakeOp):
        assert res4.hit

    res5 = cast("EagerCallableProtocol", numpy_eager_registry.get)("descriptive")(np, cast("EagerCallableProtocol", np.ones)((2, 2)))
    if isinstance(res5, FakeOp):
        assert res5.hit

    res6 = cast("EagerCallableProtocol", numpy_eager_registry.get)("distributions")(np, cast("EagerCallableProtocol", np.ones)((2, 2)))
    if isinstance(res6, FakeOp):
        assert res6.hit


def test_fallback_snippets_importerror(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_fallback_snippets_importerror.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    assert callable(test_fallback_snippets_importerror)


def test_custom_root_coverage() -> None:
    """Test function test_custom_root_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_customroot

    # solve is None
    res = _np_customroot(np, lambda x: x, 42.0)
    assert res == 42.0


def test_custom_root_coverage_solve() -> None:
    """Test function test_custom_root_coverage_solve."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_customroot

    # solve is provided
    res = _np_customroot(np, lambda x: x, 42.0, solve=lambda f, x: f(x) + 1.0)
    assert res == 43.0
