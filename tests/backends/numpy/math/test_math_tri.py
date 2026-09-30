"""Tests for test_math_tri."""

from __future__ import annotations

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


def test_math_tri_operations() -> None:
    """Test triangular and matrix operations in math_tri.

    Returns:
        None
    """
    # Tri
    assert np.array_equal(tri_mod._np_tri(np, 3, 3), np.tri(3, 3))
    assert np.array_equal(numpy_eager_registry.get("Tri")(np, 3, 3), np.tri(3, 3))

    # TrilIndices & TrilIndicesFrom
    assert np.array_equal(tri_mod._np_trilindices(np, 3)[0], np.tril_indices(3)[0])
    assert np.array_equal(numpy_eager_registry.get("TrilIndices")(np, 3)[0], np.tril_indices(3)[0])

    mat = np.ones((3, 3))
    assert np.array_equal(tri_mod._np_trilindicesfrom(np, mat)[0], np.tril_indices_from(mat)[0])
    assert np.array_equal(numpy_eager_registry.get("TrilIndicesFrom")(np, mat)[0], np.tril_indices_from(mat)[0])

    # TrimZeros
    trim_arr = np.array([0, 1, 2, 0])
    assert np.array_equal(tri_mod._np_trimzeros(np, trim_arr), np.array([1, 2]))
    assert np.array_equal(numpy_eager_registry.get("TrimZeros")(np, trim_arr), np.array([1, 2]))

    # TriuIndices & TriuIndicesFrom
    assert np.array_equal(tri_mod._np_triuindices(np, 3)[0], np.triu_indices(3)[0])
    assert np.array_equal(numpy_eager_registry.get("TriuIndices")(np, 3)[0], np.triu_indices(3)[0])

    assert np.array_equal(tri_mod._np_triuindicesfrom(np, mat)[0], np.triu_indices_from(mat)[0])
    assert np.array_equal(numpy_eager_registry.get("TriuIndicesFrom")(np, mat)[0], np.triu_indices_from(mat)[0])

    # Fromstring
    str_res = tri_mod._np_fromstring_(np, "1 2 3", sep=" ")
    assert np.array_equal(str_res, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(numpy_eager_registry.get("Fromstring")(np, "1 2 3", sep=" "), np.array([1.0, 2.0, 3.0]))

    # MatrixPower
    eye_res = tri_mod._np_linalg_matrix_power_(np, np.eye(2), 3)
    assert np.array_equal(eye_res, np.eye(2))
    assert np.array_equal(numpy_eager_registry.get("MatrixPower")(np, np.eye(2), 3), np.eye(2))

    # AsStringConfig
    cfg = tri_mod._np_asstringconfig(np, precision=5, scientific=True)
    assert isinstance(cfg, AsStringConfig)
    assert cfg.precision == 5 and cfg.scientific is True
    assert isinstance(numpy_eager_registry.get("AsStringConfig")(np, precision=5), AsStringConfig)

    # LinearOperator variations
    b_tri = tri_mod._np_linearoperatorblocklowertriangular(np)
    assert isinstance(b_tri, LinearOperatorBlockLowerTriangular)
    assert isinstance(
        numpy_eager_registry.get("LinearOperatorBlockLowerTriangular")(np),
        LinearOperatorBlockLowerTriangular,
    )

    full_op = tri_mod._np_linearoperatorfullmatrix(np)
    assert isinstance(full_op, LinearOperatorFullMatrix)
    assert isinstance(numpy_eager_registry.get("LinearOperatorFullMatrix")(np), LinearOperatorFullMatrix)

    lower_op = tri_mod._np_linearoperatorlowertriangular(np)
    assert isinstance(lower_op, LinearOperatorLowerTriangular)
    assert isinstance(
        numpy_eager_registry.get("LinearOperatorLowerTriangular")(np),
        LinearOperatorLowerTriangular,
    )

    tridiag_op = tri_mod._np_linearoperatortridiag(np)
    assert isinstance(tridiag_op, LinearOperatorTridiag)
    assert isinstance(
        numpy_eager_registry.get("LinearOperatorTridiag")(np),
        LinearOperatorTridiag,
    )

    # ConfusionMatrix
    assert tri_mod._np_confusion_matrix(np) is None
    y_t = np.array([0, 1, 2])
    y_p = np.array([0, 2, 1])
    cm1 = tri_mod._np_confusion_matrix(np, y_t, y_p)
    assert cm1 is not None and cm1.shape == (3, 3)
    cm2 = tri_mod._np_confusion_matrix(np, y_t, y_p, num_classes=4)
    assert cm2 is not None and cm2.shape == (4, 4)
    assert numpy_eager_registry.get("ConfusionMatrix")(np, y_t, y_p) is not None

    # Distributions
    assert tri_mod._np_distributions(np) is None
    dist_res = tri_mod._np_distributions(np, np.array([1.0, 2.0, 2.0, 3.0]))
    assert dist_res is not None and "counts" in dist_res and "bins" in dist_res
    assert numpy_eager_registry.get("Distributions")(np, np.array([1.0, 2.0])) is not None

    # StridedSlice
    data = np.arange(10)
    assert np.array_equal(tri_mod._np_stridedslice(np, data, [1], [8], [2]), data[1:8:2])
    assert np.array_equal(numpy_eager_registry.get("StridedSlice")(np, data, [1], [8], [2]), data[1:8:2])
