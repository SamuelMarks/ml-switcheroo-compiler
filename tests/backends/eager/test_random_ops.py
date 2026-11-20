"""Test module."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo_compiler.backends.eager.random_ops import (
    prng_key,
    rand,
    randint,
    randn,
    random_fold_in,
    random_split,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockTestStub, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyBackend(MockTestStub):
    """Test class DummyBackend."""

    def __init__(self, use_random: BackendArg = False, use_method: BackendArg = False) -> None:
        """Test function __init__.

        Args:
            use_random: Test parameter `use_random`.
            use_method: Test parameter `use_method`.
        """
        if use_random:

            class RandomMod:
                """Test class RandomMod."""

                def rand(self, *s: BackendArg) -> BackendArg:
                    """Test function rand.

                    Args:
                        s: Test parameter `s`.

                    Returns:
                        BackendArg: Computed output value from `rand`.
                    """
                    return "rand"

                def randn(self, *s: BackendArg) -> BackendArg:
                    """Test function randn.

                    Args:
                        s: Test parameter `s`.

                    Returns:
                        BackendArg: Computed output value from `randn`.
                    """
                    return "randn"

                def randint(self, low: BackendArg, h: BackendArg, size: BackendArg = None) -> BackendArg:
                    """Test function randint.

                    Args:
                        low: Test parameter `low`.
                        h: Test parameter `h`.
                        size: Test parameter `size`.

                    Returns:
                        BackendArg: Computed output value from `randint`.
                    """
                    return "randint"

            self.random = RandomMod()
        if use_method:
            self.rand = lambda *s: "rand_direct"
            self.randn = lambda *s: "randn_direct"

        self.arrays: list[tuple[BackendArg, BackendArg]] = []

    def array(self, data: BackendArg, dtype: BackendArg = None) -> BackendArg:
        """Test function array.

        Args:
            data: Test parameter `data`.
            dtype: Test parameter `dtype`.

        Returns:
            BackendArg: Computed output value from `array`.
        """
        self.arrays.append((data, dtype))
        return "array_res"


def test_random_ops() -> None:
    """Test function test_random_ops."""

    class _BareDummy(MockTestStub):
        """Empty dummy class without attributes."""

    bk_arr = DummyBackend()
    bk_no_arr = _BareDummy()

    assert prng_key(bk_arr, 42) == "array_res"
    assert bk_arr.arrays[-1] == ([0, 42], "uint32")
    assert prng_key(bk_no_arr, 42) == [0, 42]

    res1 = random_split(bk_arr, None, 2)
    assert res1 == "array_res"
    assert len(coerce_operand_list(bk_arr.arrays[-1][0])) == 2

    res2 = random_split(bk_no_arr, None, 2)
    assert len(coerce_operand_list(res2)) == 2

    assert random_fold_in(bk_arr, [1, 2], 3) == "array_res"
    assert bk_arr.arrays[-1] == ([4, 2], "uint32")
    assert random_fold_in(bk_no_arr, [1, 2], 3) == [4, 2]
    assert random_fold_in(bk_no_arr, _BareDummy(), 3) == [
        3,
        0,
    ]  # non subscriptable

    # Rand
    bk_rand1 = DummyBackend(use_random=True)
    assert rand(bk_rand1, (2, 2)) == "rand"
    bk_rand2 = DummyBackend(use_method=True)
    assert rand(bk_rand2, shape=(2, 2)) == "rand_direct"

    class DummyFallback(MockTestStub):
        """Test class DummyFallback."""

        pass

    with pytest.raises(AttributeError):
        rand(DummyFallback(), (2, 2))

    # Randn
    assert randn(bk_rand1, (2, 2)) == "randn"
    assert randn(bk_rand2, shape=(2, 2)) == "randn_direct"
    with pytest.raises(AttributeError):
        randn(DummyFallback(), (2, 2))

    # Randint
    assert randint(bk_rand1, 0, 10, (2, 2)) == "randint"
    with pytest.raises(AttributeError):
        randint(DummyFallback(), 0, 10, (2, 2))


def test_rand_coverage_auto() -> None:
    """Test function test_rand_coverage_auto."""

    class ModUniform(MockTestStub):
        """Test class ModUniform."""

        def uniform(self, size: BackendArg) -> BackendArg:
            """Test function uniform.

            Args:
                size: Test parameter `size`.

            Returns:
                BackendArg: Computed output value from `uniform`.
            """
            return "uniform_" + str(size)

    assert rand(ModUniform(), shape=(2,)) == "uniform_(2,)"

    class ModRand(MockTestStub):
        """Test class ModRand."""

        class rand_mod:
            """Test class rand_mod."""

            pass

        random = rand_mod()

        def rand(self, *args: BackendArg) -> BackendArg:
            """Test function rand.

            Args:
                args: Test parameter `args`.

            Returns:
                BackendArg: Computed output value from `rand`.
            """
            return "rand_" + str(args)

    assert rand(ModRand(), shape=(2, 3)) == "rand_(2, 3)"


def test_randn_coverage_auto() -> None:
    """Test function test_randn_coverage_auto."""

    class ModNormal(MockTestStub):
        """Test class ModNormal."""

        def normal(self, size: BackendArg) -> BackendArg:
            """Test function normal.

            Args:
                size: Test parameter `size`.

            Returns:
                BackendArg: Computed output value from `normal`.
            """
            return "normal_" + str(size)

    assert randn(ModNormal(), shape=(2,)) == "normal_(2,)"

    class ModRandn(MockTestStub):
        """Test class ModRandn."""

        class rand_mod:
            """Test class rand_mod."""

            pass

        random = rand_mod()

        def randn(self, *args: BackendArg) -> BackendArg:
            """Test function randn.

            Args:
                args: Test parameter `args`.

            Returns:
                BackendArg: Computed output value from `randn`.
            """
            return "randn_" + str(args)

    assert randn(ModRandn(), shape=(2, 3)) == "randn_(2, 3)"


def test_rand_uniform() -> None:
    """Test function test_rand_uniform."""
    mod = MagicMock()
    mod.random = MagicMock()
    mod.random.uniform = MagicMock(return_value="uniform_val")
    del mod.random.rand  # ensure it hits uniform
    assert rand(mod, shape=(2, 2)) == "uniform_val"


def test_randn_normal() -> None:
    """Test function test_randn_normal."""
    mod = MagicMock()
    mod.random = MagicMock()
    mod.random.normal = MagicMock(return_value="normal_val")
    del mod.random.randn  # ensure it hits normal
    assert randn(mod, shape=(2, 2)) == "normal_val"
