"""Tests for test_math_tri."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_tri as tri_mod
from ml_switcheroo_compiler.ops.linalg.linear_operator import (
    LinearOperatorBlockLowerTriangular,
    LinearOperatorFullMatrix,
    LinearOperatorLowerTriangular,
    LinearOperatorTridiag,
)
from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def _test_math_tri_operations_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_tri_operations.

    Args:
        _state: Shared test state dictionary.
    """
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", tri_mod._np_tri)(np, 3, 3), typing.cast("EagerCallableProtocol", np.tri)(3, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("Tri")(np, 3, 3), typing.cast("EagerCallableProtocol", np.tri)(3, 3))

    # TrilIndices & TrilIndicesFrom
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", tri_mod._np_trilindices)(np, 3), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.tril_indices)(3), "__getitem__")(0))
    )
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("TrilIndices")(np, 3), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.tril_indices)(3), "__getitem__")(0))
    )

    mat = np.ones((3, 3))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", tri_mod._np_trilindicesfrom)(np, mat), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.tril_indices_from)(mat), "__getitem__")(0))
    )
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("TrilIndicesFrom")(np, mat), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.tril_indices_from)(mat), "__getitem__")(0))
    )

    # TrimZeros
    trim_arr = np.array([0, 1, 2, 0])
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", tri_mod._np_trimzeros)(np, trim_arr), typing.cast("EagerCallableProtocol", np.array)([1, 2]))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("TrimZeros")(np, trim_arr), typing.cast("EagerCallableProtocol", np.array)([1, 2]))

    # TriuIndices & TriuIndicesFrom
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", tri_mod._np_triuindices)(np, 3), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.triu_indices)(3), "__getitem__")(0))
    )
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("TriuIndices")(np, 3), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.triu_indices)(3), "__getitem__")(0))
    )

    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", tri_mod._np_triuindicesfrom)(np, mat), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.triu_indices_from)(mat), "__getitem__")(0))
    )
    assert typing.cast("EagerCallableProtocol", np.array_equal)(
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("TriuIndicesFrom")(np, mat), "__getitem__")(0)), typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.triu_indices_from)(mat), "__getitem__")(0))
    )

    # Fromstring
    str_res = tri_mod._np_fromstring_(np, "1 2 3", sep=" ")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(str_res, typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0, 3.0]))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("Fromstring")(np, "1 2 3", sep=" "), typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0, 3.0]))

    # MatrixPower
    eye_res = tri_mod._np_linalg_matrix_power_(np, np.eye(2), 3)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(eye_res, typing.cast("EagerCallableProtocol", np.eye)(2))
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("MatrixPower")(np, typing.cast("EagerCallableProtocol", np.eye)(2), 3), typing.cast("EagerCallableProtocol", np.eye)(2))

    # AsStringConfig
    cfg = tri_mod._np_asstringconfig(np, precision=5, scientific=True)
    assert isinstance(cfg, AsStringConfig)
    getattr(_state, "update")({"cfg": cfg, "eye_res": eye_res, "mat": mat, "str_res": str_res})
    getattr(_state, "update")({"trim_arr": trim_arr})
    _state.update(locals())


def _test_math_tri_operations_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_tri_operations.

    Args:
        _state: Shared test state dictionary.
    """
    cfg = _state["cfg"]
    assert cfg.precision == 5 and bool(cfg.scientific) is True
    assert isinstance(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("AsStringConfig")(np, precision=5), AsStringConfig)

    # LinearOperator variations
    b_tri = tri_mod._np_linearoperatorblocklowertriangular(np)
    assert isinstance(b_tri, LinearOperatorBlockLowerTriangular)
    assert isinstance(
        typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("LinearOperatorBlockLowerTriangular")(np),
        LinearOperatorBlockLowerTriangular,
    )

    full_op = tri_mod._np_linearoperatorfullmatrix(np)
    assert isinstance(full_op, LinearOperatorFullMatrix)
    assert isinstance(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("LinearOperatorFullMatrix")(np), LinearOperatorFullMatrix)

    lower_op = tri_mod._np_linearoperatorlowertriangular(np)
    assert isinstance(lower_op, LinearOperatorLowerTriangular)
    assert isinstance(
        typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("LinearOperatorLowerTriangular")(np),
        LinearOperatorLowerTriangular,
    )

    tridiag_op = tri_mod._np_linearoperatortridiag(np)
    assert isinstance(tridiag_op, LinearOperatorTridiag)
    assert isinstance(
        typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("LinearOperatorTridiag")(np),
        LinearOperatorTridiag,
    )

    # ConfusionMatrix
    assert tri_mod._np_confusion_matrix(np) is None
    y_t = np.array([0, 1, 2])
    y_p = np.array([0, 2, 1])
    cm1 = tri_mod._np_confusion_matrix(np, y_t, y_p)
    assert cm1 is not None and typing.cast("EagerTensorProtocol", getattr(cm1, "shape")) == (3, 3)
    cm2 = tri_mod._np_confusion_matrix(np, y_t, y_p, num_classes=4)
    assert cm2 is not None and typing.cast("EagerTensorProtocol", getattr(cm2, "shape")) == (4, 4)
    assert typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("ConfusionMatrix")(np, y_t, y_p) is not None
    getattr(_state, "update")({"b_tri": b_tri, "cm1": cm1, "cm2": cm2, "full_op": full_op})
    getattr(_state, "update")({"lower_op": lower_op, "tridiag_op": tridiag_op, "y_p": y_p, "y_t": y_t})
    _state.update(locals())


def _test_math_tri_operations_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_tri_operations.

    Args:
        _state: Shared test state dictionary.
    """
    assert tri_mod._np_distributions(np) is None
    dist_res = tri_mod._np_distributions(np, np.array([1.0, 2.0, 2.0, 3.0]))
    assert typing.cast("EagerTensorProtocol", dist_res) is not typing.cast("EagerTensorProtocol", None) and typing.cast("EagerTensorProtocol", "counts") in typing.cast("EagerTensorProtocol", dist_res) and typing.cast("EagerTensorProtocol", "bins") in typing.cast("EagerTensorProtocol", dist_res)
    assert typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("Distributions")(np, typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0])) is not None

    # StridedSlice
    data = np.arange(10)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", tri_mod._np_stridedslice)(np, data, [1], [8], [2]), data[1:8:2])
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("StridedSlice")(np, data, [1], [8], [2]), data[1:8:2])
    getattr(_state, "update")({"data": data, "dist_res": dist_res})


def test_math_tri_operations() -> None:
    """Test triangular and matrix operations in math_tri.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_tri_operations_part_1(_state)
    _test_math_tri_operations_part_2(_state)
    _test_math_tri_operations_part_3(_state)
    assert len(_state) > 0
