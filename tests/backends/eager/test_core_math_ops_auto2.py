import numpy as np

import ml_switcheroo_compiler.backends.eager.core_math_ops as mod


def test_even_more_coverage():
    class BkSpecial:
        class linalg:
            @staticmethod
            def householder_product(*a, **k):
                return np.array([1.0])

        class fft:
            @staticmethod
            def fft2(*a, **k):
                return np.array([1.0])

            @staticmethod
            def fftfreq(*a, **k):
                return np.array([1.0])

            @staticmethod
            def fftn(*a, **k):
                return np.array([1.0])

            @staticmethod
            def fftshift(*a, **k):
                return np.array([1.0])

            @staticmethod
            def ifft(*a, **k):
                return np.array([1.0])

            @staticmethod
            def ifft2(*a, **k):
                return np.array([1.0])

            @staticmethod
            def ifftn(*a, **k):
                return np.array([1.0])

            @staticmethod
            def ifftshift(*a, **k):
                return np.array([1.0])

        class lax:
            @staticmethod
            def infeed(*a, **k):
                return np.array([1.0])

            @staticmethod
            def outfeed(*a, **k):
                return np.array([1.0])

            @staticmethod
            def pshuffle(*a, **k):
                return np.array([1.0])

            @staticmethod
            def pswapaxes(*a, **k):
                return np.array([1.0])

            @staticmethod
            def ppermute(*a, **k):
                return np.array([1.0])

            @staticmethod
            def psum_scatter(*a, **k):
                return np.array([1.0])

        class random:
            @staticmethod
            def tridiagonal(*a, **k):
                return np.array([1.0])

    bk = BkSpecial()
    arg = np.array([1.0])

    mod._accumulate_n(np, [arg, arg])

    mod._householder_product(bk, arg)
    mod._fft2(bk, arg)
    mod._fftfreq(bk, arg)
    mod._fftn(bk, arg)
    mod._fftshift(bk, arg)
    mod._ifft(bk, arg)
    mod._ifft2(bk, arg)
    mod._ifftn(bk, arg)
    mod._ifftshift(bk, arg)

    mod._infeed(bk, arg)
    mod._outfeed(bk, arg)
    mod._pshuffle(bk, arg)
    mod._pswapaxes(bk, arg)
    mod._ppermute(bk, arg)

    mod._mock_tridiagonal(bk, arg)

    mod._fftn(bk, arg)
    mod._psumscatter(bk, arg)
