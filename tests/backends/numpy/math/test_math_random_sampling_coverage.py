"""Tests for test_math_random_sampling_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_random_sampling as rand_module
from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorPermutation


def test_math_random_sampling_ops() -> None:
    """Test random sampling operations directly and through eager registry.

    Returns:
        None
    """
    # LinearOperatorPermutation
    perm_op = rand_module._np_linearoperatorpermutation(np)
    assert isinstance(perm_op, LinearOperatorPermutation)
    reg_perm = numpy_eager_registry.get("LinearOperatorPermutation")
    assert isinstance(reg_perm(np), LinearOperatorPermutation)

    # SobolSample
    sobol = rand_module._np_sobolsample(np, dim=3, num_results=10, skip=2)
    assert sobol.shape == (10, 3)
    reg_sobol = numpy_eager_registry.get("SobolSample")
    assert reg_sobol(np, dim=3, num_results=10, skip=2).shape == (10, 3)

    # RandomCategorical branches
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    cat_arg = rand_module._np_randomcategorical(np, logits, 4)
    assert cat_arg.shape == (2, 4)

    cat_kw = rand_module._np_randomcategorical(np, logits, num_samples=3)
    assert cat_kw.shape == (2, 3)

    cat_def = rand_module._np_randomcategorical(np, logits)
    assert cat_def.shape == (2, 1)

    reg_cat = numpy_eager_registry.get("RandomCategorical")
    assert reg_cat is not None

    # RandomPermutation 0-d and 1-d
    perm_0d = rand_module._np_randompermutation(np, np.array(5))
    assert len(perm_0d) == 5

    perm_1d = rand_module._np_randompermutation(np, np.array([10, 20, 30]))
    assert len(perm_1d) == 3

    reg_rand_perm = numpy_eager_registry.get("RandomPermutation")
    assert reg_rand_perm is not None

    # RandomTruncatedNormal branches
    norm_kw = rand_module._np_randomtruncatednormal(np, shape=(2, 3))
    assert norm_kw.shape == (2, 3)

    norm_arg = rand_module._np_randomtruncatednormal(np, (3, 4))
    assert norm_arg.shape == (3, 4)

    norm_none = rand_module._np_randomtruncatednormal(np)
    assert isinstance(norm_none, float)

    reg_norm = numpy_eager_registry.get("RandomTruncatedNormal")
    assert reg_norm(np, shape=(2, 2)).shape == (2, 2)

    # RandomBernoulli branches
    bern_kw = rand_module._np_randombernoulli(np, shape=(2, 3), p=0.8)
    assert bern_kw.shape == (2, 3)

    bern_arg = rand_module._np_randombernoulli(np, (3, 2), 0.4)
    assert bern_arg.shape == (3, 2)

    bern_none = rand_module._np_randombernoulli(np)
    assert isinstance(bern_none, (int, np.integer))

    reg_bern = numpy_eager_registry.get("RandomBernoulli")
    assert reg_bern(np, shape=(2, 2)).shape == (2, 2)
