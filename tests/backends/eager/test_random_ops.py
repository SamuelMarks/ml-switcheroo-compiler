"""Test module."""

from ml_switcheroo_compiler.backends.eager.random_ops import prng_key, rand, randint, randn, random_fold_in, random_split


class DummyBackend:
    def __init__(self, use_random=False, use_method=False):
        if use_random:

            class RandomMod:
                def rand(self, *s):
                    return "rand"

                def randn(self, *s):
                    return "randn"

                def randint(self, l, h, size=None):
                    return "randint"

            self.random = RandomMod()
        if use_method:
            self.rand = lambda *s: "rand_direct"
            self.randn = lambda *s: "randn_direct"

        self.arrays = []

    def array(self, data, dtype=None):
        self.arrays.append((data, dtype))
        return "array_res"


def test_random_ops():
    bk_arr = DummyBackend()
    bk_no_arr = object()

    assert prng_key(bk_arr, 42) == "array_res"
    assert bk_arr.arrays[-1] == ([0, 42], "uint32")
    assert prng_key(bk_no_arr, 42) == [0, 42]

    res1 = random_split(bk_arr, None, 2)
    assert res1 == "array_res"
    assert len(bk_arr.arrays[-1][0]) == 2

    res2 = random_split(bk_no_arr, None, 2)
    assert len(res2) == 2

    assert random_fold_in(bk_arr, [1, 2], 3) == "array_res"
    assert bk_arr.arrays[-1] == ([4, 2], "uint32")
    assert random_fold_in(bk_no_arr, [1, 2], 3) == [4, 2]
    assert random_fold_in(bk_no_arr, object(), 3) == [3, 0]  # non subscriptable

    # Rand
    bk_rand1 = DummyBackend(use_random=True)
    assert rand(bk_rand1, (2, 2)) == "rand"
    bk_rand2 = DummyBackend(use_method=True)
    assert rand(bk_rand2, shape=(2, 2)) == "rand_direct"

    import pytest

    class DummyFallback:
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
