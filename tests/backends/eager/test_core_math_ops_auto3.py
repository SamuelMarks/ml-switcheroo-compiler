import warnings

warnings.filterwarnings("ignore")
import numpy as np

import ml_switcheroo_compiler.backends.eager.core_math_ops as mod


def test_final_fftn():
    class DummyFFT:
        def fftn(self, *a, **k):
            return np.array([1.0])

    class BkSpecial:
        fft = DummyFFT()

    mod._fftn(BkSpecial(), np.array([1.0]))
    mod._fftnd(BkSpecial(), np.array([1.0]))
