"""Tests for discrete distributions."""

import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.distributions_discrete import (
    bernoulli,
    binomial,
    categorical,
    choice,
    geometric,
    multinomial,
    permutation,
    poisson,
    rademacher,
    randint,
)


def test_randint() -> None:
    """Test randint function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = randint("key", (2, 2), 0, 10, dtype=dtypes.DType.Int64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomRandint", ["key"], (2, 2), dtypes.DType.Int64, {"minval": 0, "maxval": 10})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = randint("key", (2, 2), 0, 10)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomRandint", ["key"], (2, 2), dtypes.DType.Int32, {"minval": 0, "maxval": 10})


def test_bernoulli() -> None:
    """Test bernoulli function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = bernoulli("key", p=0.8, shape=(2, 2))
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomBernoulli", ["key"], (2, 2), dtypes.DType.Bool, {"p": 0.8})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = bernoulli("key")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomBernoulli", ["key"], (), dtypes.DType.Bool, {"p": 0.5})


def test_categorical() -> None:
    """Test categorical function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = categorical("key", "logits", axis=0, shape=(2, 2))
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomCategorical", ["key"], (2, 2), dtypes.DType.Int32, {"axis": 0})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        logits_tensor = MagicMock(spec=Tensor)
        result = categorical("key", logits_tensor)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomCategorical", ["key", logits_tensor], (), dtypes.DType.Int32, {"axis": -1})


def test_permutation() -> None:
    """Test permutation function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        x_mock = MagicMock()
        x_mock.shape = (5,)
        x_mock.dtype = dtypes.DType.Float32
        result = permutation("key", x_mock, axis=1, independent=True)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomPermutation", ["key", x_mock], (5,), dtypes.DType.Float32, {"axis": 1, "independent": True})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = permutation("key", "x")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomPermutation", ["key", "x"], (), None, {"axis": 0, "independent": False})


def test_choice() -> None:
    """Test choice function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        a_mock = MagicMock()
        a_mock.dtype = dtypes.DType.Float32
        result = choice("key", a_mock, shape=(2,), replace=False, axis=1)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomChoice", ["key", a_mock], (2,), dtypes.DType.Float32, {"replace": False, "axis": 1})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        a_mock = MagicMock()
        a_mock.dtype = dtypes.DType.Float32
        p_tensor = MagicMock(spec=Tensor)
        result = choice("key", a_mock, p=p_tensor)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomChoice", ["key", a_mock, p_tensor], (), dtypes.DType.Float32, {"replace": True, "axis": 0})


def test_binomial() -> None:
    """Test binomial function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = binomial("key", "n", "p", shape=(2, 2), dtype=dtypes.DType.Int64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomBinomial", ["key"], (2, 2), dtypes.DType.Int64)

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = binomial("key", "n", "p")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomBinomial", ["key"], (), dtypes.DType.Int32)


def test_poisson() -> None:
    """Test poisson function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = poisson("key", "lam", shape=(2, 2), dtype=dtypes.DType.Int64)
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomPoisson", ["key", "lam"], (2, 2), dtypes.DType.Int64)

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = poisson("key", "lam")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomPoisson", ["key", "lam"], (), dtypes.DType.Int32)


def test_multinomial() -> None:
    """Test multinomial function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = multinomial("key", 5, "pvals", shape=(2, 2))
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomMultinomial", ["key", "pvals"], (2, 2), dtypes.DType.Int32, {"n": 5})

    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_emit_random_node") as mock_emit:
        mock_emit.return_value = "mocked_result"
        result = multinomial("key", 5, "pvals")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("RandomMultinomial", ["key", "pvals"], (), dtypes.DType.Int32, {"n": 5})


def test_geometric() -> None:
    """Test geometric function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = geometric("arg1", kwarg1="val1")
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("geometric", "arg1", kwarg1="val1")


def test_rademacher() -> None:
    """Test rademacher function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.distributions_discrete"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"
        result = rademacher("arg1", kwarg1="val1")
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("rademacher", "arg1", kwarg1="val1")
