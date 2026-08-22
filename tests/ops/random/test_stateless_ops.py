from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import (
    Algorithm,
    Generator,
    NormalConfig,
    RandomGenerationConfig,
    UniformConfig,
    create_rng_state,
    get_global_generator,
    index_shuffle,
    set_global_generator,
    stateless_beta,
    stateless_categorical,
    stateless_fold_in,
    stateless_gamma,
    stateless_parameterized_truncated_normal,
    stateless_poisson,
    stateless_random_binomial,
    stateless_random_normal,
    stateless_random_uniform,
    stateless_shuffle,
    stateless_split,
    stateless_truncated_normal,
)


def test_stateless_random_uniform(mocker):
    # We mock uniform, multiply, add, full to avoid needing backends
    mocker.patch("ml_switcheroo_compiler.random.continuous.uniform", return_value="unif")
    mocker.patch("ml_switcheroo_compiler.ops.creation.full", return_value="full")
    mocker.patch("ml_switcheroo_compiler.ops.binary.multiply", return_value="mult")
    mocker.patch("ml_switcheroo_compiler.ops.binary.add", return_value="add")

    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(eager_mode=True):
        res = stateless_random_uniform((2, 3), seed, 1.0, 5.0, "float32")
    assert res == "add"

    res2 = stateless_random_uniform((2, 3), seed, 1.0, 5.0, DType("float32"))
    assert res2 == "add"


def test_stateless_random_normal(mocker):
    mocker.patch("ml_switcheroo_compiler.random.continuous.normal", return_value="norm")
    mocker.patch("ml_switcheroo_compiler.ops.creation.full", return_value="full")
    mocker.patch("ml_switcheroo_compiler.ops.binary.multiply", return_value="mult")
    mocker.patch("ml_switcheroo_compiler.ops.binary.add", return_value="add")

    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    res = stateless_random_normal((2, 3), seed, 1.0, 5.0, "float32")
    assert res == "add"

    res2 = stateless_random_normal((2, 3), seed, 1.0, 5.0, DType("float32"))
    assert res2 == "add"


def test_stateless_random_binomial(mocker):
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete.binomial", return_value="binom")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    res = stateless_random_binomial((2, 3), seed, 10.0, 0.5, "int32")
    assert res == "binom"
    res2 = stateless_random_binomial((2, 3), seed, 10.0, 0.5, DType("int32"))
    assert res2 == "binom"


def test_stateless_truncated_normal(mocker):
    mocker.patch("ml_switcheroo_compiler.random.continuous.truncated_normal", return_value="trunc")
    mocker.patch("ml_switcheroo_compiler.ops.creation.full", return_value="full")
    mocker.patch("ml_switcheroo_compiler.ops.binary.multiply", return_value="mult")
    mocker.patch("ml_switcheroo_compiler.ops.binary.add", return_value="add")

    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    res = stateless_truncated_normal((2, 3), seed, 1.0, 5.0, "float32")
    assert res == "add"
    res2 = stateless_truncated_normal((2, 3), seed, 1.0, 5.0, DType("float32"))
    assert res2 == "add"


def test_stateless_categorical(mocker):
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete.categorical", return_value="cat")
    mocker.patch("ml_switcheroo_compiler.ops.cast", return_value="cast")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    logits = Tensor([[1.0, 2.0]], TensorConfig((1, 2), "float32", "cpu"))
    res = stateless_categorical(logits, 5, seed, "int32")
    assert res == "cast"
    res2 = stateless_categorical(logits, 5, seed, DType("int32"))
    assert res2 == "cast"

    # test no shape
    logits_no_shape = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    res3 = stateless_categorical(logits_no_shape, 5, seed, "int32")
    assert res3 == "cast"


def test_stateless_gamma(mocker):
    mocker.patch("ml_switcheroo_compiler.random.continuous.gamma", return_value="gamma")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    alpha = Tensor(2.0, TensorConfig((), "float32", "cpu"))
    res = stateless_gamma((2, 3), seed, alpha, "float32")
    assert res == "gamma"
    res2 = stateless_gamma((2, 3), seed, alpha, DType("float32"))
    assert res2 == "gamma"


def test_stateless_beta(mocker):
    mocker.patch("ml_switcheroo_compiler.random.continuous.beta", return_value="beta")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    alpha = Tensor(2.0, TensorConfig((), "float32", "cpu"))
    beta_param = Tensor(3.0, TensorConfig((), "float32", "cpu"))
    res = stateless_beta((2, 3), seed, alpha, beta_param, "float32")
    assert res == "beta"
    res2 = stateless_beta((2, 3), seed, alpha, beta_param, DType("float32"))
    assert res2 == "beta"


def test_stateless_shuffle(mocker):
    mocker.patch("ml_switcheroo_compiler.random.transformations.shuffle", return_value="shuffle")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    x = Tensor([1, 2, 3], TensorConfig((3,), "int32", "cpu"))
    res = stateless_shuffle(x, seed, 0)
    assert res == "shuffle"


def test_stateless_parameterized_truncated_normal():
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    config = RandomGenerationConfig()
    res = stateless_parameterized_truncated_normal((2,), seed, config)
    assert res.config.shape == (2,)


def test_algorithms():
    assert Algorithm.PHILOX == 1
    assert Algorithm.THREEFRY == 2
    assert Algorithm.AUTO_SELECT == 3


def test_generator():
    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = True
    try:
        g = Generator()
        assert g.state is None

        g2 = Generator.from_seed("seed123", Algorithm.PHILOX)
        assert g2.state == "seed123"

        t_norm = g.normal((2,), NormalConfig(), "float32", "my_norm")
        assert t_norm.shape == (2,)

        t_norm2 = g.normal((2,), None, "float32", "my_norm")
        assert t_norm2.shape == (2,)

        t_unif = g.uniform((3,), UniformConfig(), "int32", "my_unif")
        assert t_unif.shape == (3,)
        t_unif2 = g.uniform((3,), None, "int32", "my_unif")
        assert t_unif2.shape == (3,)
    finally:
        config.eager_mode = orig


def test_create_rng_state():
    state = create_rng_state(42)
    assert state.config.shape == (2,)
    assert state.config.dtype == DType.Int64


def test_global_generator():
    g = get_global_generator()
    assert isinstance(g, Generator)

    g2 = Generator.from_seed(999)
    set_global_generator(g2)
    assert get_global_generator() is g2
    set_global_generator(None)
    g3 = get_global_generator()
    assert isinstance(g3, Generator)


def test_index_shuffle():
    res = index_shuffle(1, "seed", 10)
    assert res == 1


def test_stateless_fold_in():
    res = stateless_fold_in("seed", "data")
    assert res == "seed"


def test_stateless_split():
    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = True
    try:
        res = stateless_split("seed", 3)
        assert res.shape == (3, 2)
    finally:
        config.eager_mode = orig


def test_stateless_poisson(mocker):
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete.poisson", return_value="poisson")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    lam = Tensor(2.0, TensorConfig((), "float32", "cpu"))
    res = stateless_poisson((2, 3), seed, lam, "int32")
    assert res == "poisson"
    res2 = stateless_poisson((2, 3), seed, lam, DType("int32"))
    assert res2 == "poisson"
