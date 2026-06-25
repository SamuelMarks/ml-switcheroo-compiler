import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random import PRNGKey, categorical, choice


def test_random_extra_coverage():
    device = Device(DeviceType.CPU, 0)

    with ConfigContext(eager_mode=True):
        key1 = PRNGKey(0)
        logits1d = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
        res1 = categorical(key1, logits1d)
        assert res1 is not None

    with ConfigContext(eager_mode=False):
        from ml_switcheroo_compiler.tracing import _tracer
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        _tracer.start_tracing()
        try:
            key2 = PRNGKey(0)
            a = Tensor(ProxyTensor("a", (3,), "int32"), TensorConfig((3,), DType.Int32, device))
            p = Tensor(ProxyTensor("p", (3,), "float32"), TensorConfig((3,), DType.Float32, device))
            res2 = choice(key2, a, shape=(10,), p=p)
            assert res2 is not None
        finally:
            _tracer.stop_tracing()


def test_permutation_eager_none():
    from unittest.mock import MagicMock

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.random import permutation

    config.eager_mode = True
    key = MagicMock()
    key.data = [0, 1]
    import numpy as np

    x = MagicMock()
    x.data = np.array([1, 2, 3])
    x.dtype = "float32"
    x.shape = (3,)

    # Test path where x has no shape/dtype getattr fallback
    class MockNoShape:
        data = np.array([1, 2, 3])

    out = permutation(key, MockNoShape())
    assert out is not None


def test_choice_eager_p():
    from unittest.mock import MagicMock

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.random import choice

    config.eager_mode = True
    key = MagicMock()
    key.data = [0, 1]
    import numpy as np

    a = MagicMock()
    a.data = np.array([1, 2, 3])
    a.dtype = "float32"

    p = MagicMock()
    p.data = np.array([0.1, 0.2, 0.7])

    out = choice(key, a, p=p)
    assert out is not None


def test_categorical_eager_2d():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.random import PRNGKey, categorical

    device = Device(DeviceType.CPU, 0)
    config.eager_mode = True
    key = PRNGKey(0)
    # 2D logits
    logits2d = Tensor(
        np.array([[1.0, 2.0], [0.5, 0.5]]), TensorConfig((2, 2), DType.Float32, device)
    )
    res = categorical(key, logits2d)
    assert res.shape == ()
    assert res.data.shape == (2,)


def test_truncated_normal_eager_rejection():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.random import PRNGKey, truncated_normal

    config.eager_mode = True
    key = PRNGKey(0)
    # Create very tight bounds so the rejection sampling loop has to run multiple times
    res = truncated_normal(key, lower=-0.0001, upper=0.0001, shape=(1000,), dtype=DType.Float32)
    assert res.shape == (1000,)


def test_all_dists_extra():  # noqa: PLR0915
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing import _tracer
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from ml_switcheroo_compiler.random import (
        PRNGKey,
        multinomial,
        beta,
        dirichlet,
        poisson,
        bernoulli,
        binomial,
        gamma,
        normal,
        uniform,
    )

    device = Device("cpu")
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            key = PRNGKey(0)
            pvals = Tensor(
                ProxyTensor("dummy_p", (2,), "float32"), TensorConfig((2,), "float32", device)
            )
            a = Tensor(ProxyTensor("dummy_a", (), "float32"), TensorConfig((), "float32", device))
            b = Tensor(ProxyTensor("dummy_b", (), "float32"), TensorConfig((), "float32", device))
            alpha = Tensor(
                ProxyTensor("dummy_alpha", (2,), "float32"), TensorConfig((2,), "float32", device)
            )
            lam = Tensor(
                ProxyTensor("dummy_lam", (), "float32"), TensorConfig((), "float32", device)
            )
            n = Tensor(ProxyTensor("dummy_n", (), "int32"), TensorConfig((), "int32", device))

            multinomial(key, 10, pvals, shape=(10,))
            beta(key, a, b, shape=(10,))
            dirichlet(key, alpha, shape=(10,))
            poisson(key, lam, shape=(10,))
            bernoulli(key, a)
            binomial(key, n, a)
            gamma(key, a, shape=(10,))
            normal(key, shape=(10,))
            uniform(key, shape=(10,))
        finally:
            _tracer.stop_tracing()

    import numpy as np

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch, MagicMock

        with patch("ml_switcheroo_compiler.random.state.get_active_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            mock_backend.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend.return_value.array.return_value = np.zeros((1,))

            key = PRNGKey(0)

            pvals = Tensor(np.array([0.5, 0.5]), TensorConfig((2,), "float32", device))
            a = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            b = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            alpha = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", device))
            lam = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            n = Tensor(np.array(10), TensorConfig((), "int32", device))

            multinomial(key, 10, pvals, shape=(10,))
            beta(key, a, b, shape=(10,))
            dirichlet(key, alpha, shape=(10,))
            poisson(key, lam, shape=(10,))
            bernoulli(key, a)
            binomial(key, n, a)
            gamma(key, a, shape=(10,))
            normal(key, shape=(10,))
            uniform(key, shape=(10,))


def test_dispatch_fallback():
    from ml_switcheroo_compiler.random.distributions_continuous import (
        ball,
        cauchy,
        f,
        gumbel,
        laplace,
        maxwell,
        pareto,
        t,
        wald,
    )
    from ml_switcheroo_compiler.random.state import (
        _dispatch_random,
        key_data,
        key_impl,
        wrap_key_data,
        clone,
        bits,
    )
    from ml_switcheroo_compiler.core.config import ConfigContext
    import pytest

    dists = [ball, cauchy, f, gumbel, laplace, maxwell, pareto, t, wald]

    with ConfigContext(eager_mode=False):
        for dist in dists:
            with pytest.raises(NotImplementedError):
                dist()
        with pytest.raises(NotImplementedError):
            _dispatch_random("key")
        with pytest.raises(NotImplementedError):
            key_data()
        with pytest.raises(NotImplementedError):
            key_impl()
        with pytest.raises(NotImplementedError):
            wrap_key_data()
        with pytest.raises(NotImplementedError):
            clone()
        with pytest.raises(NotImplementedError):
            bits()

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch, MagicMock

        with patch("ml_switcheroo_compiler.random.state.get_active_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            mock_backend.return_value.module = object()
            for dist in dists:
                with pytest.raises(NotImplementedError):
                    dist()
            with pytest.raises(NotImplementedError):
                _dispatch_random("key")
            with pytest.raises(NotImplementedError):
                key_data()
            with pytest.raises(NotImplementedError):
                key_impl()
            with pytest.raises(NotImplementedError):
                wrap_key_data()
            with pytest.raises(NotImplementedError):
                clone()
            with pytest.raises(NotImplementedError):
                bits()


def test_all_dists_extra2():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from unittest.mock import patch, MagicMock
    from ml_switcheroo_compiler.random.distributions_continuous import (
        chisquare,
        double_sided_maxwell,
        exponential,
        generalized_normal,
        loggamma,
        logistic,
        lognormal,
        multivariate_normal,
        orthogonal,
        random_gamma_p,
        rayleigh,
        triangular,
        weibull_min,
    )
    from ml_switcheroo_compiler.random.distributions_discrete import geometric, rademacher
    from ml_switcheroo_compiler.random.state import (
        key_data,
        key_impl,
        wrap_key_data,
        clone,
        bits,
        _dispatch_random,
    )

    with ConfigContext(eager_mode=True):
        with patch(
            "ml_switcheroo_compiler.random.distributions_continuous.get_active_backend"
        ) as mock_backend2:
            mock_backend2.return_value = MagicMock()
            mod = MagicMock()
            for fn in [
                "chisquare",
                "dirichlet",
                "double_sided_maxwell",
                "exponential",
                "generalized_normal",
                "loggamma",
                "logistic",
                "lognormal",
                "multivariate_normal",
                "orthogonal",
                "random_gamma_p",
                "rayleigh",
                "triangular",
                "weibull_min",
            ]:
                setattr(mod.random, fn, MagicMock(return_value="val"))
            mock_backend2.return_value.module = mod

            assert chisquare() == "val"
            assert double_sided_maxwell() == "val"
            assert exponential() == "val"
            assert generalized_normal() == "val"
            assert loggamma() == "val"
            assert logistic() == "val"
            assert lognormal() == "val"
            assert multivariate_normal() == "val"
            assert orthogonal() == "val"
            assert random_gamma_p() == "val"
            assert rayleigh() == "val"
            assert triangular() == "val"
            assert weibull_min() == "val"

        with patch(
            "ml_switcheroo_compiler.random.distributions_discrete.get_active_backend"
        ) as mock_backend3:
            mock_backend3.return_value = MagicMock()
            mod = MagicMock()
            for fn in ["geometric", "rademacher"]:
                setattr(mod.random, fn, MagicMock(return_value="val"))
            mock_backend3.return_value.module = mod

            assert geometric() == "val"
            assert rademacher() == "val"

        with patch("ml_switcheroo_compiler.random.state.get_active_backend") as mock_backend:
            mock_backend.return_value = MagicMock()
            mod = MagicMock()
            for fn in ["key_data", "key_impl", "wrap_key_data", "clone", "bits", "key"]:
                setattr(mod.random, fn, MagicMock(return_value="val"))
            mock_backend.return_value.module = mod

            assert key_data() == "val"
            assert key_impl() == "val"
            assert wrap_key_data() == "val"
            assert clone() == "val"
            assert bits() == "val"
            assert _dispatch_random("key") == "val"


def test_all_dists_extra3():
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.random.distributions_discrete import randint, categorical, choice
    from ml_switcheroo_compiler.random.state import split, fold_in
    from ml_switcheroo_compiler.random import PRNGKey
    import numpy as np

    device = Device("cpu")

    with ConfigContext(eager_mode=False):
        from ml_switcheroo_compiler.tracing import _tracer
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        _tracer.start_tracing()
        try:
            key = PRNGKey(0)
            a = Tensor(
                ProxyTensor("dummy_a", (3,), "int32"), TensorConfig((3,), DType.Int32, device)
            )
            randint(key, shape=(10,), minval=0, maxval=10)
            categorical(key, a, shape=(10,))
            choice(key, a, shape=(10,))
            split(key, num=2)
            fold_in(key, data=1)
        finally:
            _tracer.stop_tracing()

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch, MagicMock

        with (
            patch(
                "ml_switcheroo_compiler.random.distributions_discrete.get_active_backend"
            ) as mock_backend3,
            patch("ml_switcheroo_compiler.random.state.get_active_backend") as mock_backend,
        ):
            mock_backend3.return_value = MagicMock()
            mock_backend3.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend3.return_value.array.return_value = np.zeros((1,))

            mock_backend.return_value = MagicMock()
            mock_backend.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend.return_value.array.return_value = np.zeros((1,))

            key = PRNGKey(0)
            a = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

            randint(key, shape=(10,), minval=0, maxval=10)
            categorical(key, a, shape=(10,))
            choice(key, a, shape=(10,))
            split(key, num=2)
            fold_in(key, data=1)


def test_all_dists_extra4():
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.random.distributions_continuous import truncated_normal
    from ml_switcheroo_compiler.random.distributions_discrete import permutation
    from ml_switcheroo_compiler.random import PRNGKey
    import numpy as np

    device = Device("cpu")

    with ConfigContext(eager_mode=False):
        from ml_switcheroo_compiler.tracing import _tracer
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        _tracer.start_tracing()
        try:
            key = PRNGKey(0)
            a = Tensor(
                ProxyTensor("dummy_a", (3,), "int32"), TensorConfig((3,), DType.Int32, device)
            )
            truncated_normal(key, lower=-1.0, upper=1.0, shape=(10,))
            permutation(key, a)
        finally:
            _tracer.stop_tracing()

    with ConfigContext(eager_mode=True):
        from unittest.mock import patch, MagicMock

        with (
            patch(
                "ml_switcheroo_compiler.random.distributions_discrete.get_active_backend"
            ) as mock_backend3,
            patch("ml_switcheroo_compiler.random.state.get_active_backend") as mock_backend,
        ):
            mock_backend3.return_value = MagicMock()
            mock_backend3.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend3.return_value.array.return_value = np.zeros((1,))

            mock_backend.return_value = MagicMock()
            mock_backend.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend.return_value.array.return_value = np.zeros((1,))

            key = PRNGKey(0)
            a = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

            truncated_normal(key, lower=-1.0, upper=1.0, shape=(10,))
            permutation(key, a)


def test_all_dists_extra5():
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.random.distributions_continuous import beta, dirichlet
    from ml_switcheroo_compiler.random.distributions_discrete import multinomial, poisson
    import numpy as np

    device = Device("cpu")

    with ConfigContext(eager_mode=True):
        # key = PRNGKey(0)
        # Test key without tensor (int)

        pvals = Tensor(np.array([0.5, 0.5]), TensorConfig((2,), DType.Float32, device))
        a = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
        b = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
        alpha = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), DType.Float32, device))
        lam = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
        # n = Tensor(np.array(10), TensorConfig((), "int32", device))

        multinomial(0, 10, pvals, shape=(10,))
        beta(0, a, b, shape=(10,))
        dirichlet(0, alpha, shape=(10,))
        poisson(0, lam, shape=(10,))


def test_all_dists_extra6():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.random.distributions_discrete import _validate_categorical_shapes

    # Hit no-branch/no-cover logic
    # _validate_categorical_shapes with empty logits
    _validate_categorical_shapes(None, (3,))

    # Hit PRNGKey eagerly with a tensor returning seed (which wouldn't make sense eagerly)
    # but the branch exists
    with ConfigContext(eager_mode=True):
        pass  # this might be difficult to hit because PRNGKey in eager returns a Tensor.


def test_emit_random_node_eager_branch():
    from ml_switcheroo_compiler.core.config import ConfigContext
    import pytest
    from ml_switcheroo_compiler.random.state import _emit_random_node
    from ml_switcheroo_compiler.core.dtype import DType

    with ConfigContext(eager_mode=True):
        with pytest.raises(NotImplementedError):
            _emit_random_node("Test", [], (10,), DType.Float32)


def test_key_dispatch():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.random.state import key
    import pytest

    with ConfigContext(eager_mode=False):
        with pytest.raises(NotImplementedError):
            key()


def test_more_random_branches():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.random.distributions_discrete import geometric, rademacher
    from ml_switcheroo_compiler.random.distributions_continuous import (
        chisquare,
        double_sided_maxwell,
        exponential,
        generalized_normal,
        loggamma,
        logistic,
        lognormal,
        multivariate_normal,
        orthogonal,
        random_gamma_p,
        rayleigh,
        triangular,
        weibull_min,
    )
    from unittest.mock import patch, MagicMock
    import pytest

    with ConfigContext(eager_mode=True):
        with (
            patch(
                "ml_switcheroo_compiler.random.distributions_continuous.get_active_backend"
            ) as mock_backend_c,
            patch(
                "ml_switcheroo_compiler.random.distributions_discrete.get_active_backend"
            ) as mock_backend_d,
        ):
            mock_backend_c.return_value = MagicMock()
            mod_c = MagicMock()

            class DummyRandomC:
                pass

            mod_c.random = DummyRandomC()
            mock_backend_c.return_value.module = mod_c

            mock_backend_d.return_value = MagicMock()
            mod_d = MagicMock()

            class DummyRandomD:
                pass

            mod_d.random = DummyRandomD()
            mock_backend_d.return_value.module = mod_d

            with pytest.raises(NotImplementedError):
                geometric()
            with pytest.raises(NotImplementedError):
                rademacher()
            with pytest.raises(NotImplementedError):
                chisquare()
            with pytest.raises(NotImplementedError):
                double_sided_maxwell()
            with pytest.raises(NotImplementedError):
                exponential()
            with pytest.raises(NotImplementedError):
                generalized_normal()
            with pytest.raises(NotImplementedError):
                loggamma()
            with pytest.raises(NotImplementedError):
                logistic()
            with pytest.raises(NotImplementedError):
                lognormal()
            with pytest.raises(NotImplementedError):
                multivariate_normal()
            with pytest.raises(NotImplementedError):
                orthogonal()
            with pytest.raises(NotImplementedError):
                random_gamma_p()
            with pytest.raises(NotImplementedError):
                rayleigh()
            with pytest.raises(NotImplementedError):
                triangular()
            with pytest.raises(NotImplementedError):
                weibull_min()


def test_more_random_branches2():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.random.distributions_discrete import multinomial, poisson
    from ml_switcheroo_compiler.random.distributions_continuous import beta, dirichlet
    import numpy as np

    with ConfigContext(eager_mode=True):
        from ml_switcheroo_compiler.random.distributions_continuous import gamma
        from ml_switcheroo_compiler.random.distributions_discrete import binomial
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.core.device import Device

        device = Device("cpu")

        a = Tensor(np.array(1.0), TensorConfig((), "float32", device))
        b = Tensor(np.array(1.0), TensorConfig((), "float32", device))
        p = Tensor(np.array(0.5), TensorConfig((), "float32", device))
        # n = Tensor(np.array(10), TensorConfig((), "int32", device))

        beta(0, a, b, shape=(10,))
        dirichlet(
            0, Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", device)), shape=(10,)
        )
        poisson(0, a, shape=(10,))
        binomial(0, 10, p, shape=(10,))
        multinomial(
            0,
            10,
            pvals=Tensor(np.array([0.5, 0.5]), TensorConfig((2,), "float32", device)),
            shape=(10,),
        )

        gamma(0, a, shape=(10,))


def test_discrete_distributions_tracing_no_shape():
    from ml_switcheroo_compiler.random.distributions_discrete import (
        categorical,
        choice,
        poisson,
        multinomial,
    )
    from ml_switcheroo_compiler.random.state import PRNGKey
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.tracing import _tracer
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.core.device import Device

    device = Device("cpu")
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            k = PRNGKey(0)
            logits1d = Tensor(
                ProxyTensor("dummy", (2,), "float32"), TensorConfig((2,), "float32", device)
            )
            categorical(k, logits1d)

            a = Tensor(
                ProxyTensor("dummy_a", (3,), "int32"), TensorConfig((3,), DType.Int32, device)
            )
            choice(k, a)

            lam = Tensor(
                ProxyTensor("dummy_lam", (), "float32"), TensorConfig((), "float32", device)
            )
            poisson(k, lam)

            pvals = Tensor(
                ProxyTensor("dummy_p", (2,), "float32"), TensorConfig((2,), "float32", device)
            )
            multinomial(k, 10, pvals)
        finally:
            _tracer.stop_tracing()


def test_all_dists_tracing_unimplemented():
    from ml_switcheroo_compiler.random.distributions_continuous import (
        ball,
        cauchy,
        f,
        gumbel,
        laplace,
        maxwell,
        pareto,
        t,
        wald,
        chisquare,
        double_sided_maxwell,
        exponential,
        generalized_normal,
        loggamma,
        logistic,
        lognormal,
        multivariate_normal,
        orthogonal,
        random_gamma_p,
        rayleigh,
        triangular,
        weibull_min,
    )
    from ml_switcheroo_compiler.random.distributions_discrete import geometric, rademacher
    from ml_switcheroo_compiler.core.config import ConfigContext
    import pytest

    with ConfigContext(eager_mode=False):
        dists = [
            ball,
            cauchy,
            f,
            gumbel,
            laplace,
            maxwell,
            pareto,
            t,
            wald,
            chisquare,
            double_sided_maxwell,
            exponential,
            generalized_normal,
            loggamma,
            logistic,
            lognormal,
            multivariate_normal,
            orthogonal,
            random_gamma_p,
            rayleigh,
            triangular,
            weibull_min,
            geometric,
            rademacher,
        ]
        for dist in dists:
            with pytest.raises(NotImplementedError):
                dist()


def test_dirichlet_beta_branch():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.random.distributions_continuous import beta, dirichlet
    import numpy as np

    with ConfigContext(eager_mode=True):
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.core.device import Device

        device = Device("cpu")
        a = Tensor(np.array(1.0), TensorConfig((), "float32", device))
        b = Tensor(np.array(1.0), TensorConfig((), "float32", device))
        key = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", device))

        beta(key, a, b)
        dirichlet(key, Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", device)))
