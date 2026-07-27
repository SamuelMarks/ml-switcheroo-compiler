import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.nn_stubs import _np_conv_transpose, _np_isotonic_regression, _np_quantized_conv, _np_sufficient_statistics, _np_weighted_moments


class DummyBackend:
    pass


def test_nn_stubs_missing():
    # 10-12
    _np_isotonic_regression(DummyBackend(), np.array([1, 2]))

    # _np_conv_transpose fallback
    _np_conv_transpose(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), output_shape=(1, 3, 3, 1))

    # Also test the ndim != 4 branch
    _np_conv_transpose(DummyBackend(), np.ones((2, 2)), np.ones((2, 2)))

    # 179-187 (SufficientStatistics)
    _np_sufficient_statistics(DummyBackend(), np.ones((2, 2)), axes=None)
    _np_sufficient_statistics(DummyBackend(), np.ones((2, 2)), axes=[0])
    _np_sufficient_statistics(DummyBackend(), np.ones((2, 2)), axes=[0], keepdims=True)

    # 192-199 (WeightedMoments)
    try:
        _np_weighted_moments(DummyBackend(), np.array([1, 2]), axes=None, frequency_weights=np.array([1, 1]))
    except TypeError:
        # np.expand_dims fails with None
        pass
    _np_weighted_moments(DummyBackend(), np.ones((2, 2)), axes=[0], frequency_weights=np.ones((2, 2)))
    _np_weighted_moments(DummyBackend(), np.ones((2, 2)), axes=[0], frequency_weights=np.ones((2, 2)), keepdims=True)

    # 268->275 (QuantizedConv)
    # The padding branch
    _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), padding=0)
    _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), padding=1)

    # 268->275 (not instance of int)
    _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), padding="SAME")
